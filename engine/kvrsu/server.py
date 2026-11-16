"""Decision server: the RSU's decision engine as a TCP service.

The OMNeT++ RSU module (sim/src/kvrsu/RsuApp.cc) connects to this server and
sends one message per decision window.  The server answers with one decision
per request and the time the decision took.  The simulator then delivers
each decision after exactly that time, so real inference latency measured on
the GPU enters the network simulation.

Protocol (plain text, one record per line, fields separated by spaces)
---------------------------------------------------------------------
client -> server
    WINDOW <idx> <simtime> <rsu_x> <rsu_y> <rsu_cpu_hz> <rsu_backlog> <n_veh> <n_req>
    V <addr> <sumo_id> <x> <y> <speed> <heading> <cpu_hz> <backlog>      (n_veh lines)
    R <req_id> <addr> <cycles> <bytes> <deadline> <age>                 (n_req lines)
    END
server -> client
    RESULT <n_req> <t_service> <n_llm> <p> <u> <m> <ghat> <t_pred>
    D <req_id> <target> <mode> <delay>                                  (n_req lines)
    END

target: vehicle addr, -1 = execute locally, -2 = RSU edge server.
mode:   0 = decided by the policy under test, 1 = heuristic fallback.
delay:  seconds until the decision is ready (0 for fallback decisions).

Run ``python -m kvrsu.server --help`` for the options.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from typing import Dict, List, Optional, Tuple

from . import decide
from .decide import Env, TARGET_LOCAL
from .prompt import LabelSet, Request, TableManager, Vehicle, suffix_text

LLM_POLICIES = ("sequential", "batched", "prefix_cache", "kvrsu_static", "kvrsu")
ALL_POLICIES = LLM_POLICIES + ("greedy", "weighted", "rf")


class Controller:
    """Turns one window of RSU state into decisions. Independent of sockets."""

    def __init__(self, args):
        self.args = args
        self.policy = args.policy
        self.env = Env(v2v_range=args.v2v_range, v2i_range=args.v2i_range, rate_bps=args.rate_bps)
        self.log = open(args.log, "w") if args.log else None
        self.engine = self.rf = None
        self.admission = False

        if self.policy in LLM_POLICIES:
            import torch
            from .costmodel import CostModel, CostParams
            from .model import load_model
            from .serving import PrefixEngine, make_engine

            self.torch = torch
            model, encode, bos = load_model(args.model, args.device, args.dtype, args.load_4bit)
            params = CostParams.load(args.profile) if args.profile else CostParams()
            params.n_layers, params.kappa = model.n_layers, float(model.kv_bytes_per_token_layer)
            params.pipelined = not args.no_streams
            self.cost = CostModel(params)
            self.engine = make_engine(self.policy, model, self.cost, args.gpu_kv_budget_mb, not args.no_streams)
            self.labels = LabelSet(encode)
            self.encode = encode
            ordering = args.ordering or ("stability" if self.policy.startswith("kvrsu") else "arrival")
            self.tm = TableManager(encode, bos, ordering, args.resort_every)
            admission = args.admission or ("on" if self.policy == "kvrsu" else "off")
            self.admission = admission == "on" and isinstance(self.engine, PrefixEngine)
            self.max_service = args.max_service
        elif self.policy == "rf":
            import joblib
            self.rf = joblib.load(args.rf_model)
            self.rf.n_jobs = 1   # per-request inference: thread start-up would dominate

    # ---------------------------------------------------------------- window
    def handle_window(self, idx: int, now: float, env_vals: Tuple[float, float, float, float],
                      vehicles: List[Vehicle], requests: List[Request]):
        t0 = time.perf_counter()
        e = self.env
        e.rsu_x, e.rsu_y, e.rsu_cpu_hz, e.rsu_backlog = env_vals
        vehs: Dict[int, Vehicle] = {v.addr: v for v in vehicles}

        known = [r for r in requests if r.addr in vehs]
        opts = {}
        for r in known:
            r.candidates = decide.pick_candidates(r, vehs, e)
            opts[r.req_id] = decide.build_options(r, vehs, e)

        chosen: Dict[int, Tuple[int, int]] = {r.req_id: (TARGET_LOCAL, 1) for r in requests}
        stats = dict(p=0, u=0, m=0, ghat=0, t_pred=0.0, n=0)

        if self.engine is not None:
            served = self._llm(known, opts, vehicles, chosen, stats)
        elif self.policy == "rf":
            for r in known:
                proba = self.rf.predict_proba([o.feat for o in opts[r.req_id]])[:, 1]
                chosen[r.req_id] = (opts[r.req_id][int(proba.argmax())].target, 0)
            served = {r.req_id for r in known}
        else:
            for r in known:
                o = (decide.greedy(r, opts[r.req_id]) if self.policy == "greedy"
                     else decide.weighted(r, opts[r.req_id], e))
                chosen[r.req_id] = (o.target, 0)
            served = {r.req_id for r in known}

        t_service = time.perf_counter() - t0
        decisions = [(r.req_id, chosen[r.req_id][0], chosen[r.req_id][1],
                      t_service if r.req_id in served else 0.0) for r in requests]
        if self.log:
            self.log.write(json.dumps(dict(
                i=idx, t=now, rsu=list(env_vals),
                veh=[[v.addr, v.sumo_id, v.x, v.y, v.speed, v.heading, v.cpu_hz, v.backlog] for v in vehicles],
                req=[[r.req_id, r.addr, r.cycles, r.size_bytes, r.deadline, r.age, r.candidates,
                      chosen[r.req_id][0], chosen[r.req_id][1]] for r in requests],
                t_service=t_service, **stats)) + "\n")
        return decisions, t_service, len(served), stats

    def _llm(self, known, opts, vehicles, chosen, stats) -> set:
        """Serve requests with the LLM; returns the ids actually served by it."""
        table_ids, handle_of = self.tm.update(vehicles)
        suffixes = {r.req_id: self.encode(suffix_text(r, handle_of)) for r in known}
        admitted = list(known)

        if self.admission and known:
            s = max(len(x) for x in suffixes.values())
            plan = self.engine.plan(table_ids, len(known), s)
            slacks = [decide.decision_slack(r, opts[r.req_id]) for r in known]
            idx, _, _ = self.cost.admit(slacks, plan["p"], plan["u"], plan["ghat"], s,
                                        self.max_service, self.engine.spill == "adaptive")
            keep = set(idx)
            admitted = [r for i, r in enumerate(known) if i in keep]
            for i, r in enumerate(known):
                if i not in keep:
                    chosen[r.req_id] = (decide.weighted(r, opts[r.req_id], self.env).target, 1)
        if not admitted:
            return set()

        logits, st = self.engine.serve(table_ids, [suffixes[r.req_id] for r in admitted])
        stats.update({k: st[k] for k in ("p", "u", "m", "ghat", "t_pred", "n")})
        logits = logits.float()
        for row, r in enumerate(admitted):
            valid = opts[r.req_id]
            ids = self.torch.tensor([self.labels.ids[o.label] for o in valid], device=logits.device)
            chosen[r.req_id] = (valid[int(logits[row, ids].argmax())].target, 0)
        return {r.req_id for r in admitted}


# ------------------------------------------------------------------ protocol
def parse_window(lines: List[str]):
    h = lines[0].split()
    idx, now = int(h[1]), float(h[2])
    env_vals = tuple(float(x) for x in h[3:7])
    vehicles, requests = [], []
    for ln in lines[1:]:
        f = ln.split()
        if f[0] == "V":
            vehicles.append(Vehicle(int(f[1]), f[2], float(f[3]), float(f[4]), float(f[5]),
                                    float(f[6]), float(f[7]), float(f[8])))
        elif f[0] == "R":
            requests.append(Request(int(f[1]), int(f[2]), float(f[3]), float(f[4]), float(f[5]), float(f[6])))
    return idx, now, env_vals, vehicles, requests


def format_reply(decisions, t_service, n_llm, stats) -> str:
    out = [f"RESULT {len(decisions)} {t_service:.6f} {n_llm} {stats['p']} {stats['u']} "
           f"{stats['m']} {stats['ghat']} {stats['t_pred']:.6f}"]
    out += [f"D {rid} {target} {mode} {delay:.6f}" for rid, target, mode, delay in decisions]
    out.append("END")
    return "\n".join(out) + "\n"


def serve_forever(ctrl: Controller, host: str, port: int, port_file: Optional[str]) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    port = srv.getsockname()[1]
    if port_file:
        with open(port_file, "w") as f:
            f.write(str(port))
    print(f"READY {port}", flush=True)
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    rfile = conn.makefile("r")
    buf: List[str] = []
    for line in rfile:
        line = line.strip()
        if line == "BYE":
            break
        if line != "END":
            if line:
                buf.append(line)
            continue
        decisions, t_service, n_llm, stats = ctrl.handle_window(*parse_window(buf))
        conn.sendall(format_reply(decisions, t_service, n_llm, stats).encode())
        buf = []
    conn.close()
    srv.close()
    if ctrl.log:
        ctrl.log.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="kvrsu decision server")
    ap.add_argument("--policy", choices=ALL_POLICIES, required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B", help='HF name/path, or "tiny-random"')
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--profile", help="cost-model JSON from scripts/profile_engine.py")
    ap.add_argument("--gpu-kv-budget-mb", type=float, help="emulated GPU memory available for the KV cache")
    ap.add_argument("--ordering", choices=("arrival", "stability", "random"), help="slot order (default per policy)")
    ap.add_argument("--resort-every", type=int, default=20)
    ap.add_argument("--admission", choices=("on", "off"), help="deadline-aware admission (default per policy)")
    ap.add_argument("--max-service", type=float, default=float("inf"), help="upper bound on service time (s)")
    ap.add_argument("--no-streams", action="store_true", help="disable overlapped CUDA copies")
    ap.add_argument("--rf-model", help="joblib file from scripts/train_rf.py (policy rf)")
    ap.add_argument("--v2v-range", type=float, default=250.0)
    ap.add_argument("--v2i-range", type=float, default=600.0)
    ap.add_argument("--rate-bps", type=float, default=3.0e6)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    ap.add_argument("--port-file", help="write the chosen port to this file")
    ap.add_argument("--log", help="write one JSON line per window (needed to build training data)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    ctrl = Controller(args)
    serve_forever(ctrl, args.host, args.port, args.port_file)


if __name__ == "__main__":
    sys.exit(main())
