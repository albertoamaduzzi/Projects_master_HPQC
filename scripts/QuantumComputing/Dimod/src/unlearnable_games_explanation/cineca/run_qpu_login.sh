#!/bin/bash
# QPU stage directly on the login node, detached from your ssh session.
# Use it only if lrd_all_serial has no internet. The client mostly waits on the network,
# so the CPU time stays far below the login-node limit; if the process is killed anyway,
# run this script again: finished calls are skipped.
#   bash cineca/run_qpu_login.sh            # start
#   tail -f logs/qpu_login.log              # follow
source "$(dirname "$0")/env.sh"
cd "$SKGAME_PKG"
mkdir -p logs
nohup python main_simulation.py qpu --out "$SKGAME_OUT" "$@" >> logs/qpu_login.log 2>&1 &
echo "started PID $! -> logs/qpu_login.log"
