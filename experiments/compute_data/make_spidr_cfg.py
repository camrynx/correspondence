"""Write a manifest + TOML for one adaptation: finetuning.toml with the paper's two changes
(max batch length 40 s, float16). usage: make_spidr_cfg.py <wav_dir> <out_dir> <name> <init_ckpt> [max_steps]"""
import csv, sys
from pathlib import Path
import soundfile as sf
wav_dir, out, name, init = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 8000
out.mkdir(parents=True, exist_ok=True)
rows, total = [], 0
for w in sorted(wav_dir.glob("*.wav")):
    n = sf.info(w).frames
    rows.append((w.stem, str(w.resolve()), n, "en")); total += n
man = out / "manifest.csv"
with man.open("w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["fileid", "path", "num_frames", "language"]); wr.writerows(rows)
print(f"{len(rows)} clips, {total/16000/3600:.3f} h")
(out / "config.toml").write_text(f'''[data]
ssl.manifest = "{man}"
ssl.max_batch_length = 640000

[run]
model_type = "spidr_reset"
save_interval = 400
keep_latest = 1
workdir = "{out}"
wandb_mode = "offline"
wandb_project = "spidr_ft"
wandb_name = "{name}"
init_ckpt = "{init}"

[model]
freeze_step = 0
ema_start_decay = 1.0
codebook_decay = 0.8

[optimizer]
lr = 5e-6
scheduler = "constant"
max_steps = {steps}
warmup_steps = 600
hold_steps = 9400
decay_steps = 10000
dtype = "float16"

[meta_update]
task_interval = 20001

[validation]
dev.manifest = "{man}"
''')
