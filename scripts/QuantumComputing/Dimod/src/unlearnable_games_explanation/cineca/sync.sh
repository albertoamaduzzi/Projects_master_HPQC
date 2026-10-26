#!/bin/bash
# Run on YOUR machine. Moves a campaign between Leonardo and your computer, so the QPU stage
# (only HTTPS to Leap needed) can run locally while the classical stage runs on Leonardo.
#
#   export LEONARDO=<username>@login.leonardo.cineca.it
#   export REMOTE_OUT=/leonardo_work/<ACCOUNT>/skgame_results/full    # = $SKGAME_OUT on Leonardo
#   export LOCAL_OUT=results/full
#
#   bash cineca/sync.sh pull-inputs     # config.json + instances/        Leonardo -> here
#   bash cineca/sync.sh push-inputs     # config.json + instances/        here -> Leonardo
#   bash cineca/sync.sh push-anneal     # anneal/ + embeddings/ (QPU)     here -> Leonardo
#   bash cineca/sync.sh pull-results    # everything but instances/       Leonardo -> here
# Every direction only adds files (no --delete), so it is safe to repeat.
set -eo pipefail
: "${LEONARDO:?set LEONARDO=<username>@login.leonardo.cineca.it}" "${REMOTE_OUT:?set REMOTE_OUT}"
LOCAL_OUT="${LOCAL_OUT:-results/full}"
mkdir -p "$LOCAL_OUT"
case "$1" in
  pull-inputs)  rsync -av "$LEONARDO:$REMOTE_OUT/config.json" "$LEONARDO:$REMOTE_OUT/instances" "$LOCAL_OUT/" ;;
  push-inputs)  ssh "$LEONARDO" mkdir -p "$REMOTE_OUT"
                rsync -av "$LOCAL_OUT/config.json" "$LOCAL_OUT/instances" "$LEONARDO:$REMOTE_OUT/" ;;
  push-anneal)  rsync -av "$LOCAL_OUT/anneal" "$LOCAL_OUT/embeddings" "$LEONARDO:$REMOTE_OUT/" ;;
  pull-results) rsync -av --exclude instances/ "$LEONARDO:$REMOTE_OUT/" "$LOCAL_OUT/" ;;
  *) echo "usage: sync.sh pull-inputs|push-inputs|push-anneal|pull-results"; exit 1 ;;
esac
