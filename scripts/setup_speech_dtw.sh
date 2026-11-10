#!/usr/bin/env bash
# Fetch Herman Kamper's speech_dtw at the commit this code was developed against, apply the
# local Python 3 port (third_party/speech_dtw.patch), and build its Cython DTW extension.
#
#     bash scripts/setup_speech_dtw.sh
#
# The patch ports the utilities to Python 3 (cPickle, np.bool, np.mean over dict views),
# adds an "npz_dir" input format and clearer errors, and adds an optional --normalize_dims
# (per-dimension normalisation, Carlin et al. 2011; off by default) to
# utils/calculate_dtw_costs.py. The DTW itself (speech_dtw/_dtw.pyx) is untouched.
set -euo pipefail
cd "$(dirname "$0")/../third_party"
COMMIT=5fb1871fb97674ba1b4403c0242f531af4acbb9f
if [ ! -d speech_dtw ]; then
    git clone https://github.com/kamperh/speech_dtw.git
    git -C speech_dtw checkout -q "$COMMIT"
    git -C speech_dtw apply ../speech_dtw.patch
fi
cd speech_dtw
python setup.py build_ext --inplace
echo "speech_dtw ready: $(pwd)"
