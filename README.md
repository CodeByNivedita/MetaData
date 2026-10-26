# Rental Agreement Metadata Extraction

This project pulls 6 fields out of rental agreements: monthly rent, start date, end date, renewal notice (in days), Party One (owner) and Party Two (tenant). The input files are either `.docx` or scanned `.png` images.

I did not use any regex or hand-written rules for the fields. An LLM reads the document and returns JSON, and it gets a couple of solved examples from the training set to learn how the labels are written.

## My approach

I treated this as a step-by-step problem and checked the output after every step before moving on.

**1. Understand the data first.**
Before writing any model code, I opened every file and compared it with its label. This showed me two things. First, the files come in two types: clean `.docx` files and scanned `.png` images, so I needed two ways to read text. Second, the labels are not perfect. Some rows were swapped, some dates do not exist, and one file has no label at all. I wrote all of these down as I found them (they are listed further below).

**2. Get clean text out of every file.**
Every later step depends on good text, so I solved this before anything else. For `.docx`, I read both paragraphs and tables, because some agreements put the rent and names inside a table. For scanned images, I tried PaddleOCR-VL-1.6 on Kaggle's free GPU. It was very slow at first (419 seconds per page), so I switched to half precision and turned on caching, which brought it to about 12 seconds. I saved the OCR output to files so I never had to run it twice. At the end of this step I had one function, `load_document()`, that gives text for any file type.

**3. Decide how to pull out the fields.**
Rental agreements are written in very different styles. One says "Rs.3800/- per month", another says "a monthly rent of Rs.5500", and some only give a duration like "eleven months" instead of an end date. Writing regex for all of these would break on the next new file. So I chose an LLM that reads the text and fills a fixed JSON schema. I used Pydantic so the output always has exactly the 6 fields.

**4. Build a way to measure progress.**
Before trying to improve anything, I wrote `evaluate.py` to score each field separately. I used two scores: a strict one (exact match) and a lenient one (ignores case and spacing). Having both helped me see whether a miss was a real mistake or just a formatting difference.

**5. Get a baseline.**
I started with a small local model (Qwen3.5-4B through Ollama) to see how far a free, offline setup could go. Rent was decent, but names scored 0 on strict matching.

**6. Look at the mistakes, not just the score.**
Going through the wrong answers one by one taught me more than the numbers did:
- Many "wrong" names were actually right. The labels had extra spaces at the start or end, so I made strict matching trim spaces first.
- The model was adding titles like "Sri" and "Prof." that the labels do not have, so I told it to remove titles.
- My prompt said "do not use the signing date as the start date", but the labels do exactly that when no other start date is given. My instruction was fighting the data, so I changed it.
- Two files had their labels swapped. I made a corrected copy (`train_fixed.csv`) and kept the original untouched, so both scores can be compared.

**7. Switch to a bigger model when the laptop hit its limit.**
During full evaluation runs, the local model kept crashing because my laptop ran out of RAM. Instead of fighting the hardware, I moved the LLM call to Groq's free API. Only one file (`extractor.py`) had to change, because the rest of the pipeline only cares about getting text in and JSON out. The larger model also read the messy OCR text much better.

**8. Teach the model the label style with examples.**
The labels follow their own habits, and I did not want to hard-code each one. So I added a retriever: for each new file, TF-IDF finds the 2 most similar training agreements, and the model sees their correct answers before answering. To keep the score honest, a file is never allowed to be its own example. This helped with rent and notice period, but made owner names slightly worse, and I have kept that result in the table instead of hiding it.

**9. Wrap it up so others can use it.**
Finally, `pipeline.py` joins everything and writes the test predictions, and a small FastAPI app lets anyone upload a file and get the JSON back.
## How it works

```
file -> loader -> text -> retriever -> LLM with examples -> Pydantic check -> JSON
```

1. **Loading.** For `.docx` files I use python-docx and read both paragraphs and tables, because some agreements keep their details inside tables. For `.png` scans I use OCR (details below). The OCR output is saved in `outputs/ocr/`, so each image is only processed once.
2. **Retrieval.** TF-IDF finds the 2 training agreements most similar to the new file.
3. **Extraction.** The LLM gets my instructions, the 2 similar agreements along with their correct answers, and then the new agreement. It replies in JSON.
4. **Validation.** The JSON is checked against a Pydantic schema, so the output always has the same 6 fields.

| File | What it does |
|---|---|
| `src/loaders.py` | Reads docx files, runs OCR on images with a cache, and `load_document()` picks the right reader |
| `src/retriever.py` | TF-IDF search over the training documents |
| `src/extractor.py` | Prompt, LLM call through Groq, and the Pydantic schema |
| `src/pipeline.py` | `predict(file)` connects everything and writes `outputs/predictions.csv` for the test files |
| `src/evaluate.py` | Strict and lenient recall for each field, and saves every score table to `outputs/scores.md` |
| `api/main.py` | FastAPI app with a `POST /extract` endpoint |

## Why I picked these models

**OCR: PaddleOCR-VL-1.6.** It is a small vision-language model built for OCR, and it keeps the reading order of a full page well. I ran it on Kaggle's free GPU. At first one page took 419 seconds. After switching to `float16` and turning on `use_cache=True`, it came down to about 12 seconds per page.

**LLM: Groq API.** My first version ran Qwen3.5-4B locally through Ollama on CPU. It worked, but during the full evaluation my laptop kept running out of RAM and the model server crashed. So I moved to Groq, which has a free plan and is much faster. A bigger model also handles messy OCR text better. The code picks an available model on its own (for me it chose `openai/gpt-oss-120b`), and you can set a fixed one with `GROQ_MODEL` in `.env`.

**Few-shot examples instead of rules.** The labels follow some habits of their own. For example, when an agreement has no separate start date, the label uses the signing date. Instead of writing a rule for every habit like this, I show the model 2 similar agreements with their answers, so it picks up the pattern from the data.

**No leakage.** When I evaluate a file, that same file is never used as one of its own examples.

## Results

These are recall scores for each field. "Strict" means an exact match after trimming spaces at the start and end. "Lenient" also ignores capital letters and extra spaces in between. All the raw tables are in `outputs/scores.md`.

| Stage | Rent | Start | End | Notice | Party One | Party Two |
|---|---|---|---|---|---|---|
| Baseline (Qwen3.5-4B local, original labels) | 7/10 | 5/10 | 4/10 | 4/9 | 0/10 (4 lenient) | 1/10 (3 lenient) |
| Groq, zero-shot, fixed labels | 8/10 | 8/10 | 5/10 | 7/9 | 4/10 | 7/10 |
| Groq, few-shot, fixed labels | 9/10 | 8/10 | 5/10 | 8/9 | 2/10 (3 lenient) | 7/10 |
| Test set (4 files, few-shot) | 4/4 | 4/4 | 3/4 | 4/4 | 2/4 | 1/4 |

A few honest notes on these numbers:

- The biggest jump came from moving to a larger model and fixing the prompt. Start date went from 5 to 8, notice from 4 to 7, and Party Two from 1 to 7.
- Few-shot helped with numbers (rent and notice went up by one each), but it made Party One worse. Looking at the outputs, the model started copying the unusual name formats from the examples. On the train set the total strict score is 39 with and without examples, so with only 10 files I would not call this a clear win either way.
- End date is stuck at 5/10 mostly because some labels contain dates that do not exist (see below). The model will not produce those.
- On the test set, the low name scores are mainly label problems, like `.B.Kishore` with a dot in front, and `V.V.Ravi Kian`, which looks like a typo for Kiran.

## Problems I found in the data

- `54770958` and `54945838` have their labels swapped in `train.csv`. I kept the original file as it is and made a corrected copy in `data/train_fixed.csv`.
- `44737744` is a badly scanned docx. The rent and the dates are not in its text at all, so no model can get them.
- Some labels have dates that do not exist, like `31.11.2009`, `31.04.2011` and `31.02.2011`.
- `24158401` appears in both `train.csv` and `test.csv`, which is data leakage. My retriever never uses a file as an example for itself, so this does not inflate the test score.
- `46239065` is in the train folder but has no row in `train.csv`.
- Many labels have extra spaces before or after the value, which is why strict matching trims them first.
- Some test labels have typos or stray characters, like `.B.Kishore` and `V.V.Ravi Kian`.

## How to run it

Setup (Windows):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

On Linux or Mac, use `source venv/bin/activate` and `cp` instead of `copy`. Then open `.env` and paste your free API key from console.groq.com.

Running things:

```bash
python src/evaluate.py train_fixed zero   # score without examples
python src/evaluate.py train_fixed        # score with few-shot examples
python src/evaluate.py test               # score on the test files
python src/pipeline.py                    # writes outputs/predictions.csv
uvicorn api.main:app --reload             # starts the API
```

Once the API is running, open http://localhost:8000/docs to try it in the browser, or use curl:

```bash
curl -F "file=@data/test/228094620-Rental-Agreement.pdf.docx" http://localhost:8000/extract
```

## Limitations

- OCR on a new image needs a GPU along with `torch` and `transformers`. The 5 images in this dataset are already processed and saved in `outputs/ocr/`, so the rest of the project runs fine on a normal laptop. If you upload a new image to the API without a GPU, it returns a clear 503 error instead of crashing.
- The dataset is very small (10 train files and 4 test files), so one file can move a score quite a lot.
- Groq's free plan has a per-minute limit. If it is hit, the code waits 20 seconds and tries again.