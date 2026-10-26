from docx import Document
from pathlib import Path

OCR_DIR = Path("outputs/ocr")

# model ko sirf zarurat pe load karenge (bhaari hai)
model = None
processor = None


def read_docx(path):
    doc = Document(path)
    lines = []

    # normal paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            lines.append(para.text.strip())

    # kuch files me data table me hota hai
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            lines.append(" | ".join(cells))

    return "\n".join(lines)


def load_ocr_model():
    global model, processor
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    name = "PaddlePaddle/PaddleOCR-VL-1.6"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(name)


def ocr_piece(img):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "OCR:"},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:-1])


def run_ocr(path):
    from PIL import Image

    if model is None:
        print("ocr model load ho raha hai...")
        load_ocr_model()

    img = Image.open(path).convert("RGB")
    w, h = img.size
    page_h = int(w * 1.3)  # ek A4 page jitni height

    texts = []
    for top in range(0, h, page_h):
        piece = img.crop((0, top, w, min(top + page_h, h)))
        texts.append(ocr_piece(piece))
    return "\n\n".join(texts)


def read_image(path):
    path = Path(path)
    cache_file = OCR_DIR / f"{path.stem}.md"

    # pehle se ocr hua hai to wahi use karo
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    # warna model chalao aur save kar do
    text = run_ocr(path)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text


def find_file(name):
    # csv me extension nahi hai, aur kuch files ".pdf.docx" hain
    # 24158401 train.csv me hai par file test folder me hai, isliye poore data me dhoondhte hain
    for f in Path("data").rglob("*"):
        if f.name.split(".")[0] == name:
            return f
    return None


def load_document(path):
    path = Path(path)
    if path.suffix.lower() == ".docx":
        return read_docx(path)
    elif path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        return read_image(path)
    else:
        raise ValueError(f"ye file type support nahi hai: {path.suffix}")


if __name__ == "__main__":
    for f in sorted(Path("data").rglob("*")):
        if f.suffix.lower() not in [".docx", ".png"]:
            continue
        text = load_document(f)
        print(f.name, "->", len(text), "characters")