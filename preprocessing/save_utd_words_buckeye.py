import os

# Configuration
INPUT_FILE = "data/Information/Buckeye/UTD/buckeye_vad.list"
OUTPUT_DIR = "data/Information/Buckeye/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wordsUTD.list")
FRAME_SHIFT = 0.01  # 10ms per frame for standard MFCCs


def convert_frames_to_seconds(frame_str, frame_shift):
    return float(frame_str) * frame_shift


def process_buckeye_data(input_path, output_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 'w' mode automatically creates the file or overwrites it if it exists
        with open(input_path, "r") as infile, open(
            output_path, "w"
        ) as outfile:
            count = 0
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split("_")
                    if len(parts) != 3:
                        print(f"Skipping malformed line: {line}")
                        continue

                    word = parts[0]
                    speaker_file = parts[1]
                    frames_part = parts[2]

                    start_frame, end_frame = frames_part.split("-")

                    start_sec = convert_frames_to_seconds(
                        start_frame, FRAME_SHIFT
                    )
                    end_sec = convert_frames_to_seconds(end_frame, FRAME_SHIFT)

                    outfile.write(
                        f"{word} {speaker_file} {start_sec:.4f} {end_sec:.4f}\n"
                    )
                    count += 1

                except ValueError:
                    print(f"Error processing line: {line}")

        print(
            f"Successfully processed {count} lines and saved to: {output_path}"
        )

    except FileNotFoundError:
        print(
            f"Error: input file not found:\n'{os.path.abspath(input_path)}'"
        )


if __name__ == "__main__":
    process_buckeye_data(INPUT_FILE, OUTPUT_FILE)