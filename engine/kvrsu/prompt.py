"""Prompt construction: the RSU resource table and the per-request suffix.

Design rules that make KV reuse possible
----------------------------------------
1. Every vehicle is one fixed-format line (a *slot*).  Values are quantised
   so that a vehicle whose state barely changed yields the identical line.
2. Each line is tokenised on its own and the token lists are concatenated.
   A change in one slot therefore never alters the tokens of another slot.
3. The table is a prefix shared by all requests of a window; everything that
   is specific to one request goes into the suffix.
4. The order of the slots decides how much of the cached KV survives: the
   cache is valid up to the first changed slot.  ``TableManager`` keeps a
   persistent order and can sort stable vehicles to the front.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HEADER = "RSU table: id x y speed heading cpu queue\n"
CAND_LABELS = "ABCDEFGH"   # up to 8 candidate vehicles per request
LOCAL, EDGE = "L", "R"     # execute locally / on the RSU edge server
ALL_LABELS = CAND_LABELS + LOCAL + EDGE


@dataclass
class Vehicle:
    addr: int                 # network address used by the simulator
    sumo_id: str = ""
    x: float = 0.0
    y: float = 0.0
    speed: float = 0.0        # m/s
    heading: float = 0.0      # rad, direction of travel
    cpu_hz: float = 1e9       # CPU capacity in cycles/s
    backlog: float = 0.0      # seconds of queued work


@dataclass
class Request:
    req_id: int
    addr: int                 # requesting vehicle
    cycles: float             # CPU cycles
    size_bytes: float
    deadline: float           # relative deadline in s (from creation)
    age: float = 0.0          # time already spent since creation
    candidates: List[int] = field(default_factory=list)  # vehicle addrs, label order


@dataclass
class Quant:
    """Quantisation steps. Coarser steps give more reuse and less precision."""
    pos_m: float = 10.0
    speed_ms: float = 1.0
    heading_sectors: int = 8
    cpu_ghz: float = 0.5
    backlog_s: float = 0.1


def _clip(v: float, hi: int) -> int:
    return max(0, min(hi, int(v)))


def entry_text(handle: int, v: Vehicle, q: Quant) -> str:
    sector = int(round((v.heading % (2 * math.pi)) / (2 * math.pi) * q.heading_sectors)) % q.heading_sectors
    return (f"{handle:03d} x{_clip(v.x / q.pos_m, 999):03d} y{_clip(v.y / q.pos_m, 999):03d} "
            f"v{_clip(round(v.speed / q.speed_ms), 99):02d} h{sector} "
            f"c{_clip(round(v.cpu_hz / 1e9 / q.cpu_ghz), 99):02d} "
            f"q{_clip(round(v.backlog / q.backlog_s), 99):02d}\n")


def suffix_text(req: Request, handle_of: Dict[int, int]) -> str:
    cands = "".join(f" {CAND_LABELS[i]}={handle_of[a]:03d}" for i, a in enumerate(req.candidates))
    return (f"task c{_clip(round(req.cycles / 1e6), 9999):04d} b{_clip(round(req.size_bytes / 1e3), 999):03d} "
            f"d{_clip(round((req.deadline - req.age) * 10), 99):02d} from {handle_of.get(req.addr, 999):03d} "
            f"cand{cands} ; answer:")


def expected_reuse(change_probs: Sequence[float]) -> float:
    """Expected number of leading slots that stay unchanged (Proposition 1).

    E[u] = sum_k prod_{i<=k} (1 - q_i); maximal when q is non-decreasing.
    """
    total, alive = 0.0, 1.0
    for q in change_probs:
        alive *= 1.0 - q
        total += alive
    return total


class LabelSet:
    """Single-token answers. Verifies that every label is one token."""

    def __init__(self, encode: Callable[[str], List[int]]):
        self.ids: Dict[str, int] = {}
        for lab in ALL_LABELS:
            ids = encode(" " + lab)
            if len(ids) != 1:
                ids = encode(lab)
            if len(ids) != 1:
                raise ValueError(f"label {lab!r} is not a single token for this tokenizer")
            self.ids[lab] = ids[0]
        if len(set(self.ids.values())) != len(self.ids):
            raise ValueError("answer labels do not map to distinct tokens")


class TableManager:
    """Keeps the slot order and builds the token sequence of the table.

    ordering = "arrival"    slots stay in order of first appearance.
    ordering = "stability"  every ``resort_every`` windows the slots are
                            re-sorted by estimated change probability, but
                            only if the expected gain over the next period
                            exceeds the one-off cost of invalidating the cache.
    ordering = "random"     reshuffled every window (worst case; also used as
                            data augmentation when building training data).
    """

    def __init__(self, encode: Callable[[str], List[int]], bos_id: Optional[int] = None,
                 ordering: str = "arrival", resort_every: int = 20, ewma: float = 0.2,
                 quant: Optional[Quant] = None, seed: int = 0):
        assert ordering in ("arrival", "stability", "random")
        self.encode, self.ordering = encode, ordering
        self.resort_every, self.ewma = resort_every, ewma
        self.quant = quant or Quant()
        self.head_ids = ([bos_id] if bos_id is not None else []) + encode(HEADER)
        self.order: List[int] = []            # vehicle addrs in slot order
        self.handle: Dict[int, int] = {}      # addr -> 3-digit handle
        self.text: Dict[int, str] = {}        # addr -> current line
        self.q: Dict[int, float] = {}         # addr -> change probability estimate
        self._tok_cache: Dict[str, List[int]] = {}
        self._free = list(range(999, -1, -1))
        self._window = 0
        self.resorts = 0
        import random
        self._rng = random.Random(seed)

    def _tokens(self, line: str) -> List[int]:
        ids = self._tok_cache.get(line)
        if ids is None:
            if len(self._tok_cache) > 200_000:
                self._tok_cache.clear()
            ids = self._tok_cache[line] = self.encode(line)
        return ids

    def update(self, vehicles: Sequence[Vehicle]) -> Tuple[List[int], Dict[int, int]]:
        """Advance one window. Returns ``(table_token_ids, addr -> handle)``."""
        self._window += 1
        present = {v.addr: v for v in vehicles}

        for addr in [a for a in self.order if a not in present]:   # departures
            self.order.remove(addr)
            self._free.append(self.handle.pop(addr))
            self.text.pop(addr, None)
            self.q.pop(addr, None)

        for addr, v in present.items():
            if addr not in self.handle:                            # arrivals go to the tail
                self.handle[addr] = self._free.pop()
                self.order.append(addr)
                self.q[addr] = 0.5
                self.text[addr] = entry_text(self.handle[addr], v, self.quant)
                continue
            line = entry_text(self.handle[addr], v, self.quant)
            changed = line != self.text[addr]
            self.q[addr] = (1 - self.ewma) * self.q[addr] + self.ewma * (1.0 if changed else 0.0)
            self.text[addr] = line

        if self.ordering == "random":
            self._rng.shuffle(self.order)
        elif self.ordering == "stability" and self._window % self.resort_every == 0:
            self._maybe_resort()

        ids = list(self.head_ids)
        for addr in self.order:
            ids.extend(self._tokens(self.text[addr]))
        return ids, dict(self.handle)

    def _maybe_resort(self) -> None:
        cand = sorted(self.order, key=lambda a: self.q[a])
        if cand == self.order:
            return
        cur = expected_reuse([self.q[a] for a in self.order])
        new = expected_reuse([self.q[a] for a in cand])
        first_diff = next(i for i, (a, b) in enumerate(zip(self.order, cand)) if a != b)
        one_off_loss = max(0.0, cur - first_diff)          # slots recomputed once because of the move
        if (new - cur) * self.resort_every > one_off_loss:
            self.order = cand
            self.resorts += 1
