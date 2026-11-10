"""
Group utterances into extraction windows (windows.list).

The WavLM/HuBERT extractors run one forward pass per window and slice words and ABX
segments out of it, so each token gets the surrounding context. Per recording, in time
order: start a window at the next utterance and keep adding utterances while the window
(start to end of the latest utterance) is at most --max-window seconds. Windows don't
overlap, every utterance is in one window, and pauses between utterances are included.
With utterances [2,5], [7,10], [20,30], [32,35] and a 15 s maximum, the windows are
[2,10] and [20,35]. An utterance longer than --max-window is reported as an error.

Lines have the same format as utterances.list: <recording key>_<window idx> <start> <end>.
--one-per-utterance makes one window per utterance.

    python preprocessing/make_windows.py data/Information/Buckeye/utterances.list \
        data/Information/Buckeye/windows.list --max-window 90
"""

import argparse
import sys


DEFAULT_MAX_WINDOW = 90.0


def read_utterances(utterances_fn):
    """recording key -> [(seg_id, start, end), ...] in time order, keys in file order."""
    by_recording = {}
    with open(utterances_fn) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg_id, start, end = line.split()
            by_recording.setdefault(seg_id.rsplit("_", 1)[0], []).append(
                (seg_id, float(start), float(end))
            )
    for utts in by_recording.values():
        utts.sort(key=lambda u: u[1])
    return by_recording


def make_windows(utts, max_window):
    """Greedy windows over one recording's time-ordered utterances: [(start, end), ...]."""
    windows = []
    for _, start, end in utts:
        if windows and end - windows[-1][0] <= max_window:
            windows[-1][1] = end
        else:
            windows.append([start, end])
    return [tuple(w) for w in windows]


def main():
    parser = argparse.ArgumentParser(
        description="Group utterances into extraction windows (windows.list)."
    )
    parser.add_argument("utterances_fn", help="utterances.list to read")
    parser.add_argument("windows_fn", help="windows.list to write")
    parser.add_argument(
        "--max-window", type=float, default=DEFAULT_MAX_WINDOW,
        help="Maximum window length in seconds (default: %(default)s - WavLM "
             "Large's fp32 forward pass peaks at ~6.6 GiB there, leaving room "
             "on the shared 24 GB GPU)",
    )
    parser.add_argument(
        "--one-per-utterance", action="store_true",
        help="One window per utterance: windows.list is a copy of utterances.list",
    )
    args = parser.parse_args()

    by_recording = read_utterances(args.utterances_fn)

    if not args.one_per_utterance:
        too_long = [
            (seg_id, end - start)
            for utts in by_recording.values()
            for seg_id, start, end in utts
            if end - start > args.max_window
        ]
        if too_long:
            for seg_id, length in too_long:
                print(f"  {seg_id}: {length:.2f}s", file=sys.stderr)
            sys.exit(
                f"Error: {len(too_long)} utterances are longer than --max-window "
                f"{args.max_window}s and fit in no window (listed above)"
            )

    n_utts = n_windows = 0
    lengths = []
    with open(args.windows_fn, "w") as f_out:
        for key, utts in by_recording.items():
            n_utts += len(utts)
            if args.one_per_utterance:
                windows = utts  # keeps each utterance's own ID
            else:
                windows = [
                    ("{}_{:03d}".format(key, idx), start, end)
                    for idx, (start, end) in enumerate(make_windows(utts, args.max_window))
                ]
            for window_id, start, end in windows:
                f_out.write("{} {:.4f} {:.4f}\n".format(window_id, start, end))
                lengths.append(end - start)
            n_windows += len(windows)

    lengths.sort()
    print(
        "Wrote {} windows over {} utterances to {} ({}; mean {:.2f}s, median "
        "{:.2f}s, max {:.2f}s)".format(
            n_windows, n_utts, args.windows_fn,
            "one per utterance" if args.one_per_utterance
            else "max-window {}s".format(args.max_window),
            sum(lengths) / len(lengths), lengths[len(lengths) // 2], lengths[-1],
        )
    )


if __name__ == "__main__":
    main()
