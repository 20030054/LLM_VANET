#!/usr/bin/env bash
# Build the kvrsu simulation model against an existing Veins build.
#
#   VEINS_ROOT=/path/to/veins-5.3.1 ./build.sh [release|debug]
#
# Requires OMNeT++ on PATH (source <omnetpp>/setenv) and Veins built in the same mode.
# Set KVRSU_CXX to override the C++ compiler command chosen by OMNeT++, e.g. when
# clang cannot find the libstdc++ headers:
#   KVRSU_CXX="clang++ --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15"
set -euo pipefail
MODE="${1:-release}"
: "${VEINS_ROOT:?set VEINS_ROOT to the Veins source directory}"
command -v opp_makemake >/dev/null || { echo "opp_makemake not found: source <omnetpp>/setenv first" >&2; exit 1; }
[ -f "$VEINS_ROOT/src/veins/package.ned" ] || { echo "VEINS_ROOT does not look like Veins: $VEINS_ROOT" >&2; exit 1; }
VEINS_ROOT="$(cd "$VEINS_ROOT" && pwd)"
cd "$(dirname "$0")/src"
opp_makemake -f --deep -o kvrsu_sim -O out -I. -I"$VEINS_ROOT/src" -L"$VEINS_ROOT/src" '-lveins$(D)' -DVEINS_IMPORT
make -j"${JOBS:-$(nproc)}" MODE="$MODE" ${KVRSU_CXX:+CXX="$KVRSU_CXX"}
echo "built: $(pwd)/kvrsu_sim$([ "$MODE" = debug ] && echo _dbg || true)"
