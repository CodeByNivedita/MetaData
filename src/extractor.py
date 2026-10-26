import os
import json
import time
from typing import Optional

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from pydantic import BaseModel

load_dotenv()  # .env file se GROQ_API_KEY padhta hai
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Groq models badalte rehte hain, isliye naam hardcode nahi kiya
# pehle .env wala GROQ_MODEL dekhte hain, warna is list me se jo pehle mile wo
PREFERRED = ["llama-3.3-70b", "gpt-oss-120b", "llama-4", "qwen", "gpt-oss-20b", "llama"]
SKIP_WORDS = ["whisper", "guard", "tts", "playai", "orpheus", "prompt"]


def pick_model():
    if os.getenv("GROQ_MODEL"):
        return os.getenv("GROQ_MODEL")
    names = [m.id for m in client.models.list().data]
    names = [n for n in names if not any(w in n.lower() for w in SKIP_WORDS)]
    for word in PREFERRED:
        for n in names:
            if word in n.lower():
                return n
    return names[0]


MODEL = None


# model ko jawab isi format me dena hoga
class RentalInfo(BaseModel):
    agreement_value: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    renewal_notice_days: Optional[str] = None
    party_one: Optional[str] = None
    party_two: Optional[str] = None


INSTRUCTIONS = """You read rental agreements and pull out key details.
What each field means:
- agreement_value: the MONTHLY rent amount, digits only (no Rs, no commas). Never the deposit.
- start_date: the date the lease period starts, format DD.MM.YYYY. If no separate start date is given, use the execution/signing date.
- end_date: the date the lease period ends, format DD.MM.YYYY. If only a duration is given, end date = start date + duration - 1 day.
- renewal_notice_days: notice period needed to end or renew the lease, in days (1 month = 30). Digits only.
- party_one: name of the owner / lessor / landlord only.
- party_two: name of the tenant / lessee only.
For both names: leave out S/o, father's name, age and address, remove titles like Mr, Mrs, Ms, Sri, Smt, Prof, Dr, and fix obvious scan spacing errors.
If a field is truly not in the document, use null.
Reply ONLY with a JSON object with keys: agreement_value, start_date, end_date, renewal_notice_days, party_one, party_two."""


def extract_fields(text, examples=None):
    """examples = [(text, answer_dict), ...] milti-julti train files ke sahi jawab"""
    global MODEL
    if MODEL is None:
        MODEL = pick_model()
        print("model:", MODEL)

    messages = [{"role": "system", "content": INSTRUCTIONS}]
    for ex_text, ex_answer in examples or []:
        messages.append({"role": "user", "content": "Agreement text:\n\n" + ex_text})
        messages.append({"role": "assistant", "content": json.dumps(ex_answer)})
    messages.append({"role": "user", "content": "Agreement text:\n\n" + text})

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            break
        except RateLimitError:
            # free plan me per-minute limit hai, thoda ruk ke dobara
            print("rate limit, 20 sec ruk rahe hain...")
            time.sleep(20)

    data = json.loads(response.choices[0].message.content)
    data = {k: (None if v is None else str(v).strip()) for k, v in data.items()}
    return RentalInfo(**data).model_dump()  # pydantic se check: sirf 6 sahi fields


if __name__ == "__main__":
    from loaders import load_document

    path = "data/train/50070534-RENTAL-AGREEMENT.docx"
    print(json.dumps(extract_fields(load_document(path)), indent=2))
