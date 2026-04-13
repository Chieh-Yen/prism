import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("/Users/cylin/github/prism/quantization_merged_slim.csv")
TARGETS = ["|MdR|"]
PAIRS = [
    ("Omega_I", "Omega_W"),
    ("delta_I", "delta_W"),
    ("Bound_I", "Bound_W"),
]


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rankdata(values):
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k][0]] = avg_rank
        i = j
    return ranks


def pearson(x, y):
    mx = statistics.mean(x)
    my = statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sum((a - mx) ** 2 for a in x)
    den_y = sum((b - my) ** 2 for b in y)
    den = math.sqrt(den_x * den_y)
    return num / den if den else float("nan")


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def valid_pairs(rows, x_key, y_key):
    pairs = []
    for row in rows:
        x = row[x_key]
        y = row[y_key]
        if math.isnan(x) or math.isnan(y):
            continue
        pairs.append((x, y))
    return pairs


def corr(rows, metric, target, fn):
    pairs = valid_pairs(rows, metric, target)
    if len(pairs) < 2:
        return float("nan")
    x, y = zip(*pairs)
    return fn(list(x), list(y))


def compare(rows, target, scope_name):
    print(f"\n=== {scope_name} | target={target} | n={len(rows)} ===")
    wins_i = 0
    wins_w = 0
    ties = 0
    for i_key, w_key in PAIRS:
        i_s = corr(rows, i_key, target, spearman)
        w_s = corr(rows, w_key, target, spearman)
        i_p = corr(rows, i_key, target, pearson)
        w_p = corr(rows, w_key, target, pearson)

        if math.isnan(i_s) or math.isnan(w_s):
            winner = "NA"
        else:
            abs_i = abs(i_s)
            abs_w = abs(w_s)
            if abs_i > abs_w:
                winner = "I"
                wins_i += 1
            elif abs_w > abs_i:
                winner = "W"
                wins_w += 1
            else:
                winner = "tie"
                ties += 1

        print(
            f"{i_key:10s} vs {w_key:10s} | "
            f"Spearman I={i_s:+.4f} W={w_s:+.4f} | "
            f"Pearson I={i_p:+.4f} W={w_p:+.4f} | winner={winner}"
        )

    print(f"summary: I wins={wins_i}, W wins={wins_w}, ties={ties}")


def group_key(row):
    return row["target_model"], row["dataset"]


def main():
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for row in rows:
        item = dict(row)
        for key in set(TARGETS + [name for pair in PAIRS for name in pair]):
            item[key] = to_float(row.get(key, ""))
        parsed.append(item)

    grouped = defaultdict(list)
    for row in parsed:
        grouped[group_key(row)].append(row)

    for target in TARGETS:
        for model, dataset in sorted(grouped):
            compare(grouped[(model, dataset)], target, f"model={model} | dataset={dataset}")


if __name__ == "__main__":
    main()
