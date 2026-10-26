import sys
import time
import pandas as pd

from pipeline import predict
from loaders import find_file
from retriever import COLUMNS

# chalane ka tareeka:
#   python src/evaluate.py train_fixed            -> few-shot ke saath
#   python src/evaluate.py train_fixed zero        -> bina examples (baseline)


def clean(value):
    # lenient match: extra space aur capital/small ka fark ignore
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).lower()


def run(split, use_examples=True):
    truth = pd.read_csv(f"data/{split}.csv", dtype=str)  # sab text me padho, warna 90 -> 90.0 ban jata hai
    rows = []

    for _, row in truth.iterrows():
        path = find_file(row["File Name"])
        start = time.time()
        pred = predict(path, use_examples, skip_name=row["File Name"])
        print(f"{path.name} done in {round(time.time() - start)} sec")

        out = {"File Name": row["File Name"]}
        for key, col in COLUMNS.items():
            out[col] = pred.get(key)
        rows.append(out)

    preds = pd.DataFrame(rows)
    tag = "fewshot" if use_examples else "zeroshot"
    preds.to_csv(f"outputs/{split}_{tag}_predictions.csv", index=False)

    lines = [f"## {split} ({tag})", "", "| Field | Strict | Lenient |", "|---|---|---|"]
    for col in COLUMNS.values():
        strict, lenient, total = 0, 0, 0
        for t, p in zip(truth[col], preds[col]):
            if clean(t) == "":
                continue  # jahan answer hi nahi diya, wo count nahi
            total += 1
            if p is not None and not pd.isna(p) and str(p).strip() == str(t).strip():
                strict += 1
            if clean(p) == clean(t):
                lenient += 1
        lines.append(f"| {col} | {strict}/{total} | {lenient}/{total} |")

    table = "\n".join(lines)
    print("\n" + table)
    # har score table save karte hain, README me improvement dikhane ke liye
    with open("outputs/scores.md", "a", encoding="utf-8") as f:
        f.write(table + "\n\n")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train_fixed"
    use_examples = not (len(sys.argv) > 2 and sys.argv[2] == "zero")
    run(split, use_examples)
