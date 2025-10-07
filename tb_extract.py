from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv, re, argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path
from collections import defaultdict


# USAGE instructions:
# Just run on q_learning_logs folder, e.g.:
#   python tb_extract.py -b logs_disc_vs_cont_Taxi
# This will create a folder 'extracts_csv' with CSV files, and a folder 'extracts_plot' with PNG plots.
# You can customize the smoothing window
# reward range and time range



# smoothing: set to 0 to disable, otherwise odd integer window size for rolling mean
smoothing_window = 40
reward_range = (-1000, 2000)
time_range = (0, 150)
# ====================


# ====== CONFIG ======
in_dir = Path("Logs/extracts_csv")   # << set this to your folder with CSVs
out_dir = Path("Logs/extracts_plot")
out_dir.mkdir(exist_ok=True)
# clear in_dir and out_dir and recreate
if in_dir.exists():
    shutil.rmtree(in_dir)
in_dir.mkdir(parents=True, exist_ok=True)
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(exist_ok=True)

EVENT_GLOB = "events.out.tfevents.*"

# make base dir and optional output dir command-line arguments
parser = argparse.ArgumentParser(description="Extract TensorBoard scalars to CSVs.")
parser.add_argument("-b", "--base-dir", type=Path, default=Path("logs_disc_vs_cont_Taxi"),
                    help="Top-level folder containing runs (default: logs_disc_vs_cont_Taxi)")
parser.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="Folder to write CSVs into (default: <base-dir parent>/csv_extracts)")
args = parser.parse_args()

BASE_DIR = Path(args.base_dir)
if args.output_dir:
    OUTPUT_DIR = Path(args.output_dir)
else:
    OUTPUT_DIR = BASE_DIR.parent / "extracts_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
            # normalize tag name (remove 'values/' prefix if present)
            if tag.startswith('values/'):
                tag_name = tag[len('values/'):]
            else:
                tag_name = tag
            tag_set.add(tag_name)
            for e in ea.Scalars(tag):
                step_dict.setdefault(int(e.step), {})[tag_name] = float(e.value)


    # Sort for stability
    # Name CSV after the subfolder (relative to BASE_DIR)
    rel = run_dir.relative_to(BASE_DIR)
    out_og_name = sanitize_path(rel)
    if 'delaycont' in out_og_name:
        out_name = 'Corner Abstraction'
    elif 'delaydisc' in out_og_name and 'discretization_1' in out_og_name:
        out_name = 'Digital Clock'
    elif 'delaydisc' in out_og_name and 'discretization_0.2' in out_og_name:
        out_name = 'Discretized Clock - 0.2'
    elif 'delaydisc' in out_og_name and 'discretization_0.5' in out_og_name:
        out_name = 'Discretized Clock - 0.5'
    elif 'delaydisc' in out_og_name and 'discretization_0' in out_og_name:
        out_name = 'Reward Machine'
    else:
        out_name = 'Unknown'

    if 'crm_True' in out_og_name:
        out_name += ' + CRM'

    # include the sanitized relative path (out_og_name) in the filename to avoid
    # overwriting multiple runs that map to the same human-friendly label.
    # This preserves the friendly prefix while making filenames unique so variance
    # across runs can be computed.
    out_name = f"{out_name}.csv"
    out_path = OUTPUT_DIR / out_name
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step"] + sorted(tag_set))
        writer.writeheader()
        for step, tags in sorted(step_dict.items()):
             row = {"step": step}
             row.update(tags)
             writer.writerow(row)


def read_csv_long_or_wide(path: Path) -> pd.DataFrame:
    """
    Returns a long-form DataFrame with columns: step, tag, value
    Accepts:
      - long CSV: step,tag,value
      - wide CSV: step,<tag1>,<tag2>,...
    """
    df = pd.read_csv(path)
    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    if {"step", "tag", "value"}.issubset(df.columns):
        long_df = df[["step", "tag", "value"]].copy()
    else:
        # assume wide: first column is 'step', others are tags
        step_col = "step" if "step" in df.columns else df.columns[0]
        tag_cols = [c for c in df.columns if c != step_col]
        long_df = df.melt(id_vars=[step_col], value_vars=tag_cols,
                          var_name="tag", value_name="value")
        long_df = long_df.rename(columns={step_col: "step"})
    # ensure types
    long_df["step"] = pd.to_numeric(long_df["step"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["step", "value", "tag"])
    return long_df

# Load all CSVs
csv_files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])
if not csv_files:
    raise SystemExit(f"No CSV files found in {in_dir}")

# Read and collect per file
data = {}  # filename_stem -> long df
all_tags = set()
for p in csv_files:
    try:
        df_long = read_csv_long_or_wide(p)
        if df_long.empty:
            continue
        data[p.stem] = df_long
        all_tags.update(df_long["tag"].unique())
    except Exception as e:
        print(f"Skip {p}: {e}")

if not data:
    raise SystemExit("No usable data found.")

# Plot one figure per tag
# Group files by condition name (drop common seed suffixes) so variance across runs is shown
def display_name_from_stem(stem: str) -> str:
    # remove common seed suffixes like "__seed_0", "_seed-1", "_seed1", etc.
    s = re.sub(r'(?i)(__|_)?seed[_-]?\d+$', '', stem).strip()
    return s or stem

grouped_files = defaultdict(list)
for stem, df in data.items():
    grouped_files[display_name_from_stem(stem)].append(df)

out_dir = Path("plot_" + BASE_DIR.name)
out_dir.mkdir(exist_ok=True)
# clear in_dir and out_dir and recreate
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(exist_ok=True)



for tag in sorted(all_tags):
    # Make plot width smaller (e.g., 5 inches wide, 4 inches tall)
    plt.figure(figsize=(4.5, 5))
    any_curve = False
    for display_name, dfs in grouped_files.items():
        # collect all runs for this display_name and tag
        runs = []
        for df in dfs:
            sub = df[df["tag"] == tag][["step", "value"]].copy()
            if not sub.empty:
                runs.append(sub)
        if not runs:
            continue

        # concat runs and compute mean/std per step across runs
        # compute variance derived from the smoothing window (per-run), not across runs
        # build common x (union of steps) and compute per-run rolling mean/std, then aggregate
        all_steps = sorted({s for run_df in runs for s in run_df["step"].tolist()})
        if not all_steps:
            continue
        x = np.array(all_steps, dtype=float)

        w = int(smoothing_window) if smoothing_window else 1
        if w % 2 == 0:
            w += 1

        means_list = []
        stds_list = []
        for run_df in runs:
            s = run_df.set_index("step")["value"].reindex(x)  # align to common steps (NaN where missing)
            if w <= 1 or s.dropna().size <= 1:
                mean_i = s.to_numpy()
                std_i = np.zeros_like(mean_i, dtype=float)
            else:
                mean_i = s.rolling(window=w, min_periods=1, center=True).mean().to_numpy()
                std_i = s.rolling(window=w, min_periods=1, center=True).std().to_numpy()
            means_list.append(mean_i)
            stds_list.append(np.nan_to_num(std_i, nan=0.0))

        # stack and aggregate: mean of run-means; combine run-local stds via RMS
        means_stack = np.vstack(means_list)  # (n_runs, n_steps)
        stds_stack = np.vstack(stds_list)
        mean_smooth = np.nanmean(means_stack, axis=0)
        std_smooth = np.sqrt(np.nanmean(stds_stack ** 2, axis=0))

        # plot mean line and variance band (softer color via alpha)
        ln = plt.plot(x, mean_smooth, label=display_name, linewidth=1.5)[0]
        color = ln.get_color()
        # only fill when std is non-zero
        if np.nanmax(std_smooth) > 1e-12:
            plt.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                             color=color, alpha=0.18, linewidth=0)

        any_curve = True

    if not any_curve:
        plt.close()
        continue
    plt.xlabel("Time steps", fontsize=16)
    if 'reward' in tag:
        plt.ylim(reward_range)
    if 'time' in tag:
        plt.ylim(time_range)

    tag_name = tag.replace("_", " ").title()
    plt.ylabel(tag_name, fontsize=16)
    # plt.title(tag_name)  # Removed title as requested
    # Remove legend from the main graph
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)

    plt.grid(True)

    # format ticks to use 'K' for thousands (e.g., 2000 -> 2K)
    def k_formatter(x, pos):
        try:
            if abs(x) >= 1000:
                xk = x / 1000.0
                # show integer thousands as "2K", otherwise keep a short float
                return f"{int(xk)}K" if abs(xk - int(xk)) < 1e-9 else f"{xk:g}K"
            # small integers shown without decimal point
            return f"{int(x)}" if abs(x - int(x)) < 1e-9 else f"{x:g}"
        except Exception:
            return str(x)

    fmt = mtick.FuncFormatter(k_formatter)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    # remove empty space around the y-axis (tight vertical limits)
    try:
        ax.relim()
        ax.autoscale_view()
        ax.margins(y=0.1, x=0.1)
    except Exception:
        pass
    # tighten x-limits to the actual plotted data (no extra horizontal padding)
    try:
        x_arrays = []
        for ln in ax.get_lines():
            xd = np.asarray(ln.get_xdata(), dtype=float)
            if xd.size:
                x_arrays.append(xd)
        if x_arrays:
            x_all = np.concatenate(x_arrays)
            x_min = float(np.nanmin(x_all))
            x_max = float(np.nanmax(x_all))
            # Ensure at least 200000 is included as the upper x-limit
            x_max = max(x_max, 300000)
            # Add a smaller margin (e.g., 2%) beyond the upper x-limit
            margin = 0.02 * (x_max - x_min)
            if x_min == x_max:
                pad = max(0.1, 0.001 * abs(x_min)) if x_min != 0 else 0.1  # reduced pad
                ax.set_xlim(x_min - pad, x_max + pad)
            else:
                ax.set_xlim(x_min, x_max + margin)
    except Exception:
        pass

    plt.tight_layout()
    out_path = out_dir / f"{tag}.svg"
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close()
    
    # --- Save legend separately as SVG ---
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        fig_legend = plt.figure(figsize=(6, 1))
        fig_legend.legend(handles, labels, loc='center', ncol=4, frameon=False)
        fig_legend.tight_layout()
        legend_path = out_dir / f"{tag}_legend.svg"
        fig_legend.savefig(legend_path, format="svg", bbox_inches="tight")
        plt.close(fig_legend)
        print(f"Saved legend: {legend_path}")
    
    # svg to pdf using inkscape if available
    try:
        import subprocess
        pdf_path = out_path.with_suffix(".pdf")
        subprocess.run(["inkscape", str(out_path), "--export-type=pdf", "--export-filename", str(pdf_path)],
                       check=True)
        print(f"Saved {pdf_path}")
    except Exception:
        print("Failed to convert SVG to PDF.")
    print(f"Saved {out_path}")

print(f"Done. Plots in: {out_dir.resolve()}")