#!/bin/bash
# Which nodes reach the D-Wave cloud?  Prints one line per place.
source "$(dirname "$0")/env.sh"
probe='curl -s -o /dev/null -m 15 -w "%{http_code}" https://cloud.dwavesys.com/sapi/ || echo TIMEOUT'
echo "login node        : $(bash -c "$probe")"
echo "lrd_all_serial    : $(srun -A "$SKGAME_ACCOUNT" -p lrd_all_serial -t 00:02:00 -c 1 bash -c "$probe" 2>&1 | tail -1)"
echo "$SKGAME_PARTITION : $(srun -A "$SKGAME_ACCOUNT" -p "$SKGAME_PARTITION" -t 00:02:00 -c 1 bash -c "$probe" 2>&1 | tail -1)"
echo "(an HTTP code such as 200/401/404 = reachable; TIMEOUT or 000 = no internet there)"
