"""
Build utterances.list and words.list for Xitsonga from its forced alignments.

Each Xitsonga recording is a single utterance, so each becomes one segment covering the
whole audio file. Words are kept if they have at least 5 characters and last at least
0.5 s.

    <speaker>_<file>_<seg_idx> <start> <end>      (utterances.list)
    <word> <file> <start> <end>                   (words.list)

    python preprocessing/save_utterances_and_words_xitsonga.py \
        data/Information/Buckeye_tsonga/xitsonga.wrd \
        data/Information/Buckeye_tsonga/utterances.list data/Information/Buckeye_tsonga/words.list
"""

import argparse
from os import path
import soundfile as sf


DEFAULT_AUDIO_DIR = "data/Datasets/Buckeye-Xitsonga"


def get_speaker(file_label):
    """ "nchlt_tso_001m_0007" -> "001m"."""
    return file_label.rsplit("_", 2)[-2]


def fa_to_segments(fa_fn, audio_dir):
    """(file_label, 0, duration) per recording, in order of first appearance; durations
    are read from the audio."""
    recordings = []
    seen = set()

    with open(fa_fn, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterance = line.split()[0]
            if utterance not in seen:
                seen.add(utterance)
                recordings.append(utterance)

    segments = []
    for file_label in recordings:
        wav_fn = path.join(audio_dir, file_label + ".wav")
        if not path.exists(wav_fn):
            raise SystemExit(
                "Error: no audio for {} at {} - pass --audio-dir if the "
                "recordings live elsewhere".format(file_label, wav_fn)
                )
        info = sf.info(wav_fn)
        segments.append((file_label, 0.0, info.frames / info.samplerate))

    return segments


def fa_to_words(fa_fn):
    """(word, file_label, start, end) for words of >= 5 characters and >= 0.5 s, in file order."""
    words = []

    with open(fa_fn, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterance, start_token, end_token, token_label = line.split()

            if token_label in ["SIL", "SPN"]:
                continue

            start_token = float(start_token)
            end_token = float(end_token)

            # Round to the millisecond to avoid floating-point misses at 0.5 s.
            duration = round(end_token - start_token, 3)
            word_length = len(token_label)

            # Apply filters: length >= 5 and duration >= 0.5 seconds
            if word_length >= 5 and duration >= 0.5:
                words.append((token_label, utterance, start_token, end_token))

    return words


def write_words_list(words, words_fn):
    """Write <word> <file_label> <start> <end> lines."""
    with open(words_fn, "w") as f_out:
        for word, utterance, start, end in words:
            f_out.write(
                "{} {} {:.4f} {:.4f}\n".format(word, utterance, start, end)
            )


def write_list_file(segments, list_fn):
    """Write segments with a running index per recording."""
    seg_counter = {}  # file_label -> next segment index

    with open(list_fn, "w") as f_out:
        for file_label, start, end in segments:
            speaker = get_speaker(file_label)
            idx = seg_counter.get(file_label, 0)
            seg_counter[file_label] = idx + 1

            seg_id = "{}_{}_{:03d}".format(speaker, file_label, idx)

            f_out.write("{} {:.4f} {:.4f}\n".format(seg_id, start, end))


def main():
    parser = argparse.ArgumentParser(
        description="Convert the Xitsonga forced alignment file into a "
        ".list file of speech segments and a .list file of individual "
        "words."
    )
    parser.add_argument("fa_fn", help="Path to the forced alignment file")
    parser.add_argument(
        "segments_list_fn", help="Path to write the segments .list file to"
    )
    parser.add_argument(
        "words_list_fn",
        nargs="?",
        default="data/Information/Buckeye_tsonga/words.list",
        help="Path to write the words .list file to "
        "(default: data/Information/Buckeye_tsonga/words.list)",
    )
    parser.add_argument(
        "--audio-dir",
        default=DEFAULT_AUDIO_DIR,
        help="Directory holding the .wav recordings, read for their durations "
        "(default: {})".format(DEFAULT_AUDIO_DIR),
    )
    args = parser.parse_args()

    segments = fa_to_segments(args.fa_fn, args.audio_dir)
    write_list_file(segments, args.segments_list_fn)
    print("Wrote {} segments to {}".format(len(segments), args.segments_list_fn))

    words = fa_to_words(args.fa_fn)
    write_words_list(words, args.words_list_fn)
    print("Wrote {} words to {}".format(len(words), args.words_list_fn))


if __name__ == "__main__":
    main()
