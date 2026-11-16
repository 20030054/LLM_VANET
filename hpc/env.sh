# Site configuration. EDIT THIS FILE ONCE, then every job script sources it.
#
#   source hpc/env.sh
#
# 1. Modules: names differ between clusters. Find yours with `module avail`.
#    Needed: a C++17 compiler (GCC >= 9 or Clang), bison, flex, Python 3.10-3.12, CUDA.
# module purge
# module load gcc/12 python/3.11 cuda/12.1

# 2. Paths (all inside the project; nothing needs root).
export KVRSU_HOME="${KVRSU_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TOOLS="${TOOLS:-$KVRSU_HOME/tools}"              # OMNeT++, Veins and the Python venv live here
export OMNETPP_ROOT="$TOOLS/omnetpp-6.1"
export VEINS_ROOT="$TOOLS/veins-veins-5.3.1"
export RESULTS="${RESULTS:-$KVRSU_HOME/results}"
export PYTHON="$TOOLS/venv/bin/python"

# 3. Tool environments.
[ -f "$OMNETPP_ROOT/setenv" ] && source "$OMNETPP_ROOT/setenv" -q >/dev/null 2>&1
[ -x "$PYTHON" ] && export SUMO_HOME="$("$PYTHON" -c 'import sumo; print(sumo.SUMO_HOME)' 2>/dev/null)"
[ -n "${SUMO_HOME:-}" ] && export PATH="$SUMO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$VEINS_ROOT/src:${LD_LIBRARY_PATH:-}"

# 4. Models. Compute nodes are usually offline: download on the login node (docs/HPC_SETUP.md, step 5).
export HF_HOME="${HF_HOME:-$KVRSU_HOME/models/hf_cache}"
export HF_HUB_OFFLINE=1
export BASE_MODEL="$KVRSU_HOME/models/llama32-1b"            # downloaded base model
export MODEL="${MODEL:-$KVRSU_HOME/models/llama32-1b-kvrsu}" # fine-tuned model used by the experiments
export RF_MODEL="$KVRSU_HOME/models/rf.joblib"
export PROFILE="$RESULTS/profile.json"
export DEVICE="${DEVICE:-cuda}"
