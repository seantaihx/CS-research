
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_list(x):
    # Sometimes a single dict is stored instead of a list of dicts
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def extract_tasks(data):
    # Top-level can be a dict with "tasklist" or a list of such dicts.
    if isinstance(data, dict) and "tasklist" in data:
        return data["tasklist"]
    if isinstance(data, list):
        # Some files may wrap multiple workload entries; collect all tasklists.
        tasks = []
        for item in data:
            if isinstance(item, dict) and "tasklist" in item:
                tasks.extend(item["tasklist"])
        return tasks
    return []

def compute_time_zero(tasks):
    starts = [t.get("start_time") for t in tasks if isinstance(t.get("start_time"), (int, float))]
    if not starts:
        raise ValueError("No numeric 'start_time' values found in tasklist.")
    return min(starts)

def gather_nodes(tasks):
    """Return sorted unique node names across all tasks."""
    nodes = set()
    for t in tasks:
        for n in ensure_list(t.get("nodes", [])):
            name = None
            if isinstance(n, dict):
                name = n.get("node_name") or n.get("name") or n.get("hostname")
            if name:
                nodes.add(name)
    return sorted(nodes)

def plot_gantt(tasks, out_path, title=None):
    t0 = compute_time_zero(tasks)
    node_names = gather_nodes(tasks)
    if not node_names:
        # Fallback: use task ids as rows
        node_names = [f"task_{t.get('task_id', i)}" for i, t in enumerate(tasks)]

    # Map node name -> y position index
    y_index = {name: i for i, name in enumerate(node_names)}

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(node_names))))

    # Track whether we've already added a legend entry per task
    seen_task_label = set()

    for i, task in enumerate(tasks):
        start = task.get("start_time")
        finish = task.get("finish_time")
        if not isinstance(start, (int, float)) or not isinstance(finish, (int, float)):
            # Skip tasks without valid times
            continue

        start_rel = start - t0
        duration = (finish - t0) - start_rel  # == finish - start

        label = task.get("app_name") or f"task_{task.get('task_id', i)}"
        nodes = ensure_list(task.get("nodes", []))

        # If there are no node entries, draw the bar on a pseudo-row for this task
        if not nodes:
            row = y_index.setdefault(label, len(y_index))
            ax.broken_barh([(start_rel, duration)], (row - 0.4, 0.8),
                           label=None if label in seen_task_label else label)
            seen_task_label.add(label)
            continue

        for n in nodes:
            if not isinstance(n, dict):
                continue
            node_name = n.get("node_name") or n.get("name") or n.get("hostname")
            if not node_name:
                continue
            row = y_index.get(node_name)
            if row is None:
                # Unknown node slipped in; append dynamically
                row = len(y_index)
                y_index[node_name] = row
                node_names.append(node_name)

            ax.broken_barh([(start_rel, duration)], (row - 0.4, 0.8),
                           label=None if label in seen_task_label else label)
            seen_task_label.add(label)

    ax.set_xlabel("Seconds since earliest start_time (s)")
    ax.set_yticks(range(len(y_index)))
    ax.set_yticklabels([name for name, _ in sorted(y_index.items(), key=lambda kv: kv[1])])
    if title:
        ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    # Only show legend if there are few enough entries
    if seen_task_label and len(seen_task_label) <= 20:
        ax.legend(loc="upper right", fontsize=8, title="Tasks")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return str(out_path)

def main():
    parser = argparse.ArgumentParser(description="Plot horizontal bar chart (Gantt) from HPC workload tasklist JSON.")
    parser.add_argument("json_path", help="Path to JSON file containing 'tasklist'.")
    parser.add_argument("--out", default="gantt.png", help="Output image path (default: gantt.png)")
    parser.add_argument("--title", default=None, help="Optional chart title")
    args = parser.parse_args()

    data = load_json(args.json_path)
    tasks = extract_tasks(data)
    if not tasks:
        raise SystemExit("No 'tasklist' found in the provided JSON.")

    out_file = plot_gantt(tasks, args.out, title=args.title)
    print(f"Saved figure to {out_file}")

if __name__ == "__main__":
    main()
