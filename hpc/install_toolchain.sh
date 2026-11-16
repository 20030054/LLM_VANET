#!/usr/bin/env bash
# Install OMNeT++ 6.1, Veins 5.3.1, SUMO 1.25.0 and the Python environment into $TOOLS (no root needed).
# Run ONCE on a login node (needs internet):   bash hpc/install_toolchain.sh
# Takes 20-40 minutes, mostly compiling OMNeT++. Safe to re-run: finished steps are skipped.
#
# INET is NOT required: the scenario uses the IEEE 802.11p stack that ships with Veins.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$TOOLS"; cd "$TOOLS"
JOBS="${JOBS:-8}"

echo "== Python environment =="
if [ ! -x "$PYTHON" ]; then
    python3 -m venv "$TOOLS/venv"
    "$PYTHON" -m pip install --upgrade pip
    # PyTorch: pick the wheel index that matches the cluster's CUDA version, e.g. cu121 or cu124.
    "$PYTHON" -m pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu121}"
    "$PYTHON" -m pip install -r "$KVRSU_HOME/requirements.txt"
fi
export SUMO_HOME="$("$PYTHON" -c 'import sumo; print(sumo.SUMO_HOME)')"
export PATH="$SUMO_HOME/bin:$PATH"
sumo --version | head -1

echo "== OMNeT++ 6.1 (command-line only: no Qtenv, no OSG) =="
if [ ! -x "$OMNETPP_ROOT/bin/opp_run" ]; then
    [ -f omnetpp-6.1.0-linux-x86_64.tgz ] || curl -L -O https://github.com/omnetpp/omnetpp/releases/download/omnetpp-6.1.0/omnetpp-6.1.0-linux-x86_64.tgz
    tar xzf omnetpp-6.1.0-linux-x86_64.tgz
    cd "$OMNETPP_ROOT"
    # OMNeT++'s own Python helpers
    "$PYTHON" -m pip install -r python/requirements.txt || true
    source setenv -q
    ./configure WITH_QTENV=no WITH_OSG=no WITH_OSGEARTH=no PREFER_CLANG=no
    make -j"$JOBS" MODE=release
    cd "$TOOLS"
fi
source "$OMNETPP_ROOT/setenv" -q

echo "== Veins 5.3.1 =="
if [ ! -f "$VEINS_ROOT/src/libveins.so" ]; then
    [ -f veins-5.3.1.zip ] || curl -L -o veins-5.3.1.zip https://github.com/sommer/veins/archive/refs/tags/veins-5.3.1.zip
    unzip -q -o veins-5.3.1.zip
    cd "$VEINS_ROOT"
    ./configure
    make -j"$JOBS" MODE=release
    cd "$TOOLS"
fi

echo "== kvrsu simulation model =="
VEINS_ROOT="$VEINS_ROOT" JOBS="$JOBS" "$KVRSU_HOME/sim/build.sh" release

echo "== engine unit tests (CPU, tiny model) =="
cd "$KVRSU_HOME" && "$PYTHON" -m pytest engine/tests -q
echo "Toolchain ready. Next: docs/HPC_SETUP.md, step 5 (download the model)."
