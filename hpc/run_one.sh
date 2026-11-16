#!/usr/bin/env bash
# Run ONE simulation: start the decision server, run OMNeT++/Veins/SUMO against it, stop the server.
#
#   run_one.sh POLICY VEHICLES SPEED SEED [TAG]
#
#   POLICY    sequential | batched | prefix_cache | kvrsu_static | kvrsu | greedy | weighted | rf
#   VEHICLES  number of vehicles (= vehicles/km^2 with the default 1 km^2 grid)
#   SPEED     speed limit in m/s
#   SEED      repetition number (OMNeT++ seed set, SUMO seed, trip seed)
#   TAG       optional label appended to the output name (for variants)
#
# Environment (see hpc/env.sh):
#   KVRSU_HOME   repository root                         (default: parent of this script)
#   VEINS_ROOT   Veins source directory                  (required)
#   PYTHON       Python interpreter with the engine deps (default: python3)
#   MODEL        model name or path                      (default: meta-llama/Llama-3.2-1B)
#   DEVICE       cuda | cpu                              (default: cuda)
#   RESULTS      output directory                        (default: $KVRSU_HOME/results)
#   SIM_TIME     simulated seconds                       (default: 150)
#   PROFILE      cost-model JSON, passed to LLM policies (optional)
#   RF_MODEL     joblib file, passed to policy rf        (required for rf)
#   SERVER_ARGS  extra arguments for kvrsu.server        (e.g. "--gpu-kv-budget-mb 64 --profile p.json")
#   SIM_ARGS     extra arguments for the simulation      (e.g. "--*.rsu[0].appl.computeScale=4")
#   SCEN_ARGS    extra arguments for make_scenario.py    (e.g. "--junction priority")
set -euo pipefail
[ $# -ge 4 ] || { sed -n 2,22p "$0"; exit 1; }
POLICY=$1; N=$2; SPEED=$3; SEED=$4; TAG=${5:-}

KVRSU_HOME="${KVRSU_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${VEINS_ROOT:?set VEINS_ROOT}"
PYTHON="${PYTHON:-python3}"; MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"; DEVICE="${DEVICE:-cuda}"
RESULTS="${RESULTS:-$KVRSU_HOME/results}"; SIM_TIME="${SIM_TIME:-150}"
NAME="${POLICY}${TAG:+-$TAG}_n${N}_v${SPEED}_s${SEED}"
OUT="$RESULTS/$NAME"; mkdir -p "$OUT"
SCEN="$KVRSU_HOME/sim/simulations/scenarios/n${N}_v${SPEED}_s${SEED}${SCEN_TAG:+_$SCEN_TAG}"

# 1. scenario (generated once, reused by all policies so they see the same traffic)
if [ ! -f "$SCEN/scenario.env" ]; then
    TMP="$SCEN.tmp.$$"
    # shellcheck disable=SC2086
    "$PYTHON" "$KVRSU_HOME/scenario/make_scenario.py" --out "$TMP" --vehicles "$N" --speed "$SPEED" --seed "$SEED" ${SCEN_ARGS:-}
    mv -T "$TMP" "$SCEN" 2>/dev/null || rm -rf "$TMP"   # another job may have created it meanwhile
fi
# shellcheck disable=SC1091
source "$SCEN/scenario.env"

# 2. decision server
rm -f "$OUT/port" "$OUT/meta.json"
case "$POLICY" in
    rf) SERVER_ARGS="--rf-model ${RF_MODEL:?set RF_MODEL for policy rf} ${SERVER_ARGS:-}" ;;
    greedy|weighted) ;;
    *) [ -n "${PROFILE:-}" ] && SERVER_ARGS="--profile $PROFILE ${SERVER_ARGS:-}" ;;
esac
# shellcheck disable=SC2086
PYTHONPATH="$KVRSU_HOME/engine${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m kvrsu.server --policy "$POLICY" \
    --model "$MODEL" --device "$DEVICE" --port-file "$OUT/port" --log "$OUT/windows.jsonl" ${SERVER_ARGS:-} \
    > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1800); do
    [ -s "$OUT/port" ] && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "decision server died:"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 1
done
[ -s "$OUT/port" ] || { echo "decision server did not start within 30 min"; exit 1; }
PORT=$(cat "$OUT/port")

# 3. simulation
cd "$KVRSU_HOME/sim/simulations"
# shellcheck disable=SC2086
"$KVRSU_HOME/sim/src/kvrsu_sim" -u Cmdenv -c Base -n "$KVRSU_HOME/sim/src:$VEINS_ROOT/src/veins" \
    --seed-set="$SEED" --sim-time-limit="${SIM_TIME}s" --result-dir="$OUT" \
    --output-scalar-file="$OUT/run.sca" \
    "--*.playgroundSizeX=${PLAYGROUND}m" "--*.playgroundSizeY=${PLAYGROUND}m" \
    "--*.rsu[0].mobility.x=$RSU_X" "--*.rsu[0].mobility.y=$RSU_Y" \
    "--*.manager.configFile=\"$SCEN/sim.sumo.cfg\"" "--*.manager.seed=$SEED" \
    "--*.rsu[0].appl.serverPort=$PORT" "--*.rsu[0].appl.logPrefix=\"$OUT/run\"" \
    ${SIM_ARGS:-} > "$OUT/sim.log" 2>&1 || { echo "simulation failed:"; tail -30 "$OUT/sim.log"; exit 1; }

# 4. the server exits when the simulation closes the connection
for _ in $(seq 30); do kill -0 $SERVER_PID 2>/dev/null || break; sleep 1; done
cat > "$OUT/meta.json" <<META
{"policy": "$POLICY", "tag": "$TAG", "vehicles": $N, "speed": $SPEED, "seed": $SEED,
 "area_km2": $AREA_KM2, "sim_time": $SIM_TIME, "model": "$MODEL",
 "server_args": "${SERVER_ARGS:-}", "sim_args": "${SIM_ARGS:-}"}
META
echo "done: $OUT"
