import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from loaders import load_document, find_file

# csv ke column names (assignment me "Aggrement" spelling hi hai)
COLUMNS = {
    "agreement_value": "Aggrement Value",
    "start_date": "Aggrement Start Date",
    "end_date": "Aggrement End Date",
    "renewal_notice_days": "Renewal Notice (Days)",
    "party_one": "Party One",
    "party_two": "Party Two",
}

EXAMPLE_CHARS = 2500  # example ka text chhota rakhte hain, taaki prompt bahut lamba na ho


class Retriever:
    def __init__(self, csv="data/train_fixed.csv"):
        self.df = pd.read_csv(csv, dtype=str).fillna("")
        self.names = list(self.df["File Name"])
        self.texts = [load_document(find_file(n)) for n in self.names]
        self.vec = TfidfVectorizer(stop_words="english").fit(self.texts)
        self.matrix = self.vec.transform(self.texts)

    def answer(self, i):
        # csv ki row ko model ke json format me badlo
        row = self.df.iloc[i]
        return {key: (row[col].strip() or None) for key, col in COLUMNS.items()}

    def similar(self, text, skip_name=None, k=2):
        scores = cosine_similarity(self.vec.transform([text]), self.matrix)[0]
        examples = []
        for i in scores.argsort()[::-1]:
            # leakage se bachne ke liye: jis file ko test kar rahe, wo khud example nahi banegi
            if self.names[i] == skip_name:
                continue
            examples.append((self.texts[i][:EXAMPLE_CHARS], self.answer(i)))
            if len(examples) == k:
                break
        return examples


if __name__ == "__main__":
    r = Retriever()
    text = load_document(find_file("50070534-RENTAL-AGREEMENT"))
    for ex_text, ans in r.similar(text, skip_name="50070534-RENTAL-AGREEMENT"):
        print(ans)
