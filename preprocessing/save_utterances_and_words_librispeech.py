"""
Build utterances.list and words.list for LibriSpeech from one or more forced-alignment
(.wrd) files, merged into one pair of lists (split membership comes from Splits/).

Recordings are split into utterances at every SIL/SPN token. Words are kept if they have
at least 5 characters and last at least 0.5 s (Carlin et al., 2011).

    <speaker>_<file_label>_<seg_idx> <start> <end>      (utterances.list)
    <word> <file_label> <start> <end>                   (words.list)

    python preprocessing/save_utterances_and_words_librispeech.py \
        --fa data/Information/LibriSpeech/librispeech_train-clean-5.wrd \
        --fa data/Information/LibriSpeech/librispeech_dev-clean.wrd \
        data/Information/LibriSpeech/utterances.list data/Information/LibriSpeech/words.list
"""

import argparse


DEFAULT_MAX_UTTERANCE = 30.0


def get_speaker(file_label):
    """ "1272-128104-0000" -> "1272"."""
    return file_label.split("-")[0]


def split_long_utterance(words, max_utterance):
    """Split a run of words at its longest internal silence until every piece is at
    most max_utterance seconds (rarely needed for LibriSpeech)."""
    if len(words) == 1 or words[-1][1] - words[0][0] <= max_utterance:
        return [words]

    split_at, largest_gap = 0, -1.0
    for i in range(len(words) - 1):
        gap = words[i + 1][0] - words[i][1]
        if gap > largest_gap:
            largest_gap, split_at = gap, i

    return (
        split_long_utterance(words[: split_at + 1], max_utterance)
        + split_long_utterance(words[split_at + 1:], max_utterance)
    )


def fa_to_segments(fa_fn, max_utterance):
    """(file_label, start, end) per utterance: runs of words between SIL/SPN tokens."""
    runs_by_file = {}  # file_label -> [[(start, end), ...], ...]; insertion-ordered
    current_file = None
    current_run = []

    def flush():
        if current_run and current_file is not None:
            runs_by_file[current_file].append(current_run)

    with open(fa_fn, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterance, start_token, end_token, token_label = line.split()

            if utterance != current_file:
                flush()
                current_file = utterance
                current_run = []
                runs_by_file.setdefault(utterance, [])

            if token_label in ["SIL", "SPN"]:
                flush()
                current_run = []
                continue

            current_run.append((float(start_token), float(end_token)))
        flush()

    segments = []
    for file_label, runs in runs_by_file.items():
        for run in runs:
            for piece in split_long_utterance(run, max_utterance):
                segments.append((file_label, piece[0][0], piece[-1][1]))

    return segments


def fa_to_words(fa_fn):
    """(word, file_label, start, end) for words of >= 5 characters and >= 0.5 s."""
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

            if word_length >= 5 and duration >= 0.5:
                words.append((token_label, utterance, start_token, end_token))

    return words


def write_words_list(words, words_fn):
    """Write words to a .list file: <word> <utterance> <start> <end>"""
    with open(words_fn, "w") as f_out:
        for word, utterance, start, end in words:
            f_out.write(
                "{} {} {:.4f} {:.4f}\n".format(word, utterance, start, end)
            )


def write_list_file(segments, list_fn):
    """Write utterances with a running index per recording."""
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
        description="Convert one or more LibriSpeech forced alignment files "
        "into a single combined .list file of speech segments and a single "
        "combined .list file of individual words."
    )
    parser.add_argument(
        # Repeatable flag rather than nargs="+", which would swallow the positionals.
        "--fa", dest="fa_fns", action="append", required=True,
        help="forced-alignment file; repeat for each subset. All are merged into the "
        "same output files, which are overwritten, not appended to.",
    )
    parser.add_argument(
        "segments_list_fn", help="Path to write the combined segments .list file to"
    )
    parser.add_argument(
        "words_list_fn",
        nargs="?",
        default="data/Information/LibriSpeech/words.list",
        help="Path to write the combined words .list file to "
        "(default: data/Information/LibriSpeech/words.list)",
    )
    parser.add_argument(
        "--max-utterance", type=float, default=DEFAULT_MAX_UTTERANCE,
        help="Safety-net maximum utterance length (seconds); longer ones are "
        "split at their largest internal silence. Not expected to trigger "
        "on real LibriSpeech data - see module docstring (default: %(default)s)",
    )
    args = parser.parse_args()

    all_segments = []
    all_words = []
    for fa_fn in args.fa_fns:
        segments = fa_to_segments(fa_fn, args.max_utterance)
        words = fa_to_words(fa_fn)
        print("{}: {} segments, {} words".format(fa_fn, len(segments), len(words)))
        all_segments.extend(segments)
        all_words.extend(words)

    write_list_file(all_segments, args.segments_list_fn)
    lengths = [end - start for _, start, end in all_segments]
    print(
        "Wrote {} segments to {} (max-utterance {}s; "
        "mean {:.2f}s, median {:.2f}s, max {:.2f}s)".format(
            len(all_segments), args.segments_list_fn,
            args.max_utterance, sum(lengths) / len(lengths),
            sorted(lengths)[len(lengths) // 2], max(lengths),
        )
    )

    write_words_list(all_words, args.words_list_fn)
    print("Wrote {} words to {}".format(len(all_words), args.words_list_fn))


if __name__ == "__main__":
    main()
