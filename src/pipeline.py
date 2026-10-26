import sys
import pandas as pd
from pathlib import Path

from loaders import load_document
from extractor import extract_fields
from retriever import Retriever, COLUMNS

retriever = None


def predict(path, use_examples=True, skip_name=None):
    """ek file -> 6 fields ka dict"""
    global retriever
    text = load_document(path)
    examples = []
    if use_examples:
        if retriever is None:
            retriever = Retriever()
        examples = retriever.similar(text, skip_name=skip_name)
    return extract_fields(text, examples)


def predict_folder(folder="data/test", out="outputs/predictions.csv"):
    rows = []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in [".docx", ".png", ".jpg", ".jpeg"]:
            continue
        name = f.name.split(".")[0]
        pred = predict(f, skip_name=name)
        row = {"File Name": name}
        for key, col in COLUMNS.items():
            row[col] = pred[key]
        rows.append(row)
        print(name, "done")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved:", out)


if __name__ == "__main__":
    predict_folder(sys.argv[1] if len(sys.argv) > 1 else "data/test")
