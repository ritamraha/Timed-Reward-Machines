from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv, re

# Top-level folder containing all Q-learning runs (with many subfolders)
BASE_DIR = Path("q_learning_logs")
EVENT_GLOB = "events.out.tfevents.*"

def sanitize_path(rel_path: Path) -> str:
    """Turn a relative path like 'grid/lr_0.1/seed_0' into a safe filename."""
    s = rel_path.as_posix().replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

# Find all subfolders that contain at least one event file
run_dirs = sorted({p.parent for p in BASE_DIR.rglob(EVENT_GLOB)})
tag_set = set()
for run_dir in run_dirs:
    # Gather all scalars from all event files under this subfolder
    print('----------------------------')
    print('TF dir:', run_dir)
    step_dict = {}
    for event_file in sorted(run_dir.glob(EVENT_GLOB)):
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            #remove 'values/' prefix if present
            if tag.startswith('values/'):
                tag_name = tag[len('values/'):]
            tag_set.add(tag_name)
            for e in ea.Scalars(tag):
                step_dict.setdefault(int(e.step), {})[tag_name] = float(e.value)

    

    # Sort for stability
    # Name CSV after the subfolder (relative to BASE_DIR)
    rel = run_dir.relative_to(BASE_DIR)
    out_name = sanitize_path(rel) + ".csv"
    out_path = BASE_DIR / out_name
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step"] + sorted(tag_set))
        writer.writeheader()
        for step, tags in step_dict.items():
            row = {"step": step}
            row.update(tags)
            writer.writerow(row)