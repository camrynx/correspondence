#!/usr/bin/env bash
# Re-time WavLM term extraction on the training split only (sizes/full/box, sandbox copy).
# The timed pipeline run also passed over devpart2 and zs, which hold no discovered terms.
#
#     bash experiments/compute_data/retime_extract.sh
#
# Waits until nobody else is on the GPU, so the timing is clean. Writes results/retime_extract.txt.
set -euo pipefail
EXP=$(cd "$(dirname "$0")" && pwd)
BOX=$EXP/sizes/full/box
PY=${PY:-python}
LOG=$EXP/logs/full_retime_extract.log
OUT=$EXP/results/retime_extract.txt
PEAK=$EXP/results/.retime_gpu

ok=0
while (( ok < 2 )); do
    if [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then ok=$((ok+1)); else ok=0; fi
    (( ok < 2 )) && sleep 60
done
echo "[$(date '+%F %T')] GPU clear, starting" | tee "$LOG"

trap 'kill "${SAMPLER:-}" 2>/dev/null || true' EXIT

: > "$PEAK"
( while sleep 2; do
    echo "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> "$PEAK"
  done ) &
SAMPLER=$!

cd "$BOX"
/usr/bin/time -v -o "$OUT.time" "$PY" features/extract_ssl.py --model wavlm --corpus buckeye --pairs dusted --splits devpart1 >> "$LOG" 2>&1

{
    grep -E "Elapsed|User time|System time|Maximum resident" "$OUT.time"
    echo "GPU peak MiB: $(sort -n "$PEAK" | tail -1 | cut -d' ' -f1)"
    echo "Most GPU processes seen: $(cut -d' ' -f2 "$PEAK" | sort -n | tail -1)"
    tr '\r' '\n' < "$LOG" | grep -E "^Wrote|Speakers: 100%" | tail -2
} | tee "$OUT"
rm -f "$OUT.time" "$PEAK"
echo "RETIME DONE"
