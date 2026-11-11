<!-- TODO: authors, venue, and links to the paper and released checkpoints -->
# Correspondence on SSL Features

Code for the paper **Lightweight Adaptation of Self-Supervised Features using Discovered Spoken-Term Pairs**.

We train correspondence autoencoders (cAEs) on MFCC, WavLM Large and HuBERT Large
features. The frame pairs come from four sources: gold word pairs from forced alignments
(full supervision), and three that need no supervision:

- **UTD**: word pairs from classic unsupervised term discovery (ZRTools)
- **DUSTED**: word pairs from DUSTED discovered terms
- **Poor Man's UTD**: every frame paired with its nearest frame from another speaker

Each cAE is trained with the usual MSE loss, and also with a contrastive (InfoNCE) loss
from a random initialisation (the "NCE cAE" in the paper, `ccae` in the code). We
evaluate with same-different average precision and triphone ABX on English (Buckeye),
Xitsonga (NCHLT) and French (LibriVox).

## Setup

```bash
conda env create -f environment.yml
conda activate cae
bash scripts/setup_speech_dtw.sh
```

The last step clones [speech_dtw](https://github.com/kamperh/speech_dtw), applies our
Python 3 patch (`third_party/speech_dtw.patch`) and builds it. PyTorch is pinned to
CUDA 12.8 in `environment.yml`; change the index URL there for other setups.

## Data

Corpora should be arranged as:

```
data/
  Datasets/
    Buckeye/<speaker>/<recording>.wav                  e.g. s02/s0201a.wav
    Buckeye-Xitsonga/nchlt_tso_<speaker>_<n>.wav
    FrenchDatasets/french/train/<speaker>_<cond>.wav
  Information/
    Buckeye/   Buckeye_tsonga/   ZS17French/
```

Each folder in `Information/` holds:

- `Splits/*_speakers.list`: the speakers in each split (Buckeye devpart1/devpart2/zs,
  Xitsonga dev/test, French train/test)
- `utterances.list` and `words.list`: voice-active segments and gold word tokens
  (`preprocessing/save_utterances_and_words_xitsonga.py` for Xitsonga)
- `windows.list`: extraction windows of up to 90 s (`preprocessing/make_windows.py`)
- `wordsUTD.list`: ZRTools terms, Buckeye and Xitsonga only (used for comparing to older paper results) (`preprocessing/save_utd_words_*.py`)
- `UTDWordsDUSTED.txt`: DUSTED terms, from the DUSTED pipeline
- `abx_items/<split>.item`: triphone ABX items (`evaluation/build_abx_item_file.py`, from
  the `.phn` phone alignments)

<!-- TODO: the scripts that built utterances.list / words.list for Buckeye and French
     are not in this repository; add them, or publish the lists themselves. -->

## Usage

`scripts/reproduce_paper.sh` runs everything behind the paper's tables. For a single
setting, e.g. WavLM on Buckeye with DUSTED pairs:

```bash
# features (WavLM layer 21), normalised with training-split statistics
python features/extract_ssl.py --model wavlm --corpus buckeye
python features/normalise.py --features wavlm --corpus buckeye
python features/extract_ssl.py --model wavlm --corpus buckeye --pairs dusted
python features/normalise.py --features wavlm --corpus buckeye --pairs dusted

# DTW alignments for all same-term pairs
python training/align.py --features wavlm --corpus buckeye --pairs dusted --n-cpus 8

# train (each run also encodes the gold words of every split)
python training/train_cae.py --features wavlm --corpus buckeye --pairs dusted
python training/train_ccae.py --features wavlm --corpus buckeye --pairs dusted

# AP and ABX on the test split
python evaluate.py --buckeye --wavlm --dusted --cae --ccae
```

The scripts share `--features mfcc|wavlm|hubert` (`--model` for `extract_ssl.py`),
`--corpus buckeye|xitsonga|french` and `--pairs gold|utd|dusted|poorman`. MFCCs come
from `features/extract_mfcc.py` and need no normalisation step. Poor Man's UTD only needs
the gold features. Training logs to Weights & Biases (`WANDB_MODE=disabled` to turn it off).

Outputs go to `data/Features/`, `data/Alignments/`, `models/` and `results/`.

`features/extras/` and `training/extras/` have experiments that aren't in the paper
(LibriSpeech, phone pairs, SpidR features, cross-lingual normalisation), and
`experiments/` has the drivers for the amount-of-data and ccAE runs.

## How we extract features
The way we extract features is important as it largely can determine downstream performance. 
With too little context (audio seen during feature extraction), models (HuBERT and WavLM) perform poorly, and with too much context, the compute required is too much to fit on a GPU.
We have three files which are important for the way we get features:
- `windows.list` = Defines the extraction window -- the chunk of speech that is given to models such as WavLM and HuBERT to extract features from.
- `utterances.list` = Defines continuous speech regions. This list is only used in DUSTED to find discovered terms, and its definition changes per dataset.
- `words.list` = Defines where words are. We feature-slice words from the features obtained using windows.list to get ground-truth words to evaluate the models against.

### How utterances are defined, per dataset:
- Buckeye: A new utterance starts at every gap between consecutive words, however short. A gap is a SIL/SPN token or any untranscribed stretch, which in Buckeye is mostly the interviewer. Each utterance runs from its first word's start to its last word's end.
- Xitsonga: Each recording is already an utterance.
- French: Stretch of non-SIL phones (from french.phn) - we only use first 18 mins of relative, and M03_N is excluded (mostly silence).

### How do we generate windows.list for every dataset:

Windows are built greedily from `utterances.list`. A window starts at the first unassigned utterance and keeps adding the next utterances while it stays at most `W` long. The first utterance that doesn't fit starts a new window. Because windows always break between utterances, every word, phone and utterance falls fully inside one window.

For example, with `W = 15 s`, utterances `[2, 5]`, `[7, 10]`, `[20, 30]`, `[32, 35]` give the windows `[2, 10]` and `[20, 35]`.

We use `W = 90 s`. We chose this as it is the longest window we tested that can process on an 8 GB GPU: on our machine, WavLM Large peaks at 6.6 GiB at 90 s and runs out of memory at 240 s.

## Citation

To do

## Acknowledgements

We use Herman Kamper's [speech_dtw](https://github.com/kamperh/speech_dtw), Nicol
Visser's [samediff](https://github.com/nicolvisser/samediff) and
[fastabx](https://github.com/bootphon/fastabx). The contrastive loss follows
[globalphone_awe_pytorch](https://github.com/christiaanjacobs/globalphone_awe_pytorch)
(Jacobs et al., SLT 2021).

## License

MIT
