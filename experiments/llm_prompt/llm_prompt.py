"""E17: smarter LLM corrector, prompt experiment (Kaggle GPU script).

The LLM only proposes fixes here; every reply is saved raw. Filtering and scoring happen
locally (score.py), so several filters can be compared on one run without the GPU.

Runs (model x prompt), each over every unique sentence in DATA:
  4b_P0   typhoon2.1-gemma3-4b, the E10 prompt (baseline)
  4b_P1   typhoon2.1-gemma3-4b, the new context-rich system prompt + more examples
  (12b dropped: 4b is enough, per the user)

Output: /kaggle/working/replies_<run>.json  {sentence_id: reply_text}
DATA is embedded by build.py: {"sentences": {id: text}, ...}. It holds ASR hypotheses only, no references.
"""
import json
import os
import subprocess
import sys
import time

DATA = None  # filled in by build.py

OUT = "/kaggle/working" if os.path.isdir("/kaggle") else "results/llm_prompt"

P0 = (
    "You check Thai speech-recognition output for spelling mistakes.\n"
    "Find words that are misspelled or are not real Thai words, and give the correct spelling.\n"
    "Rules:\n"
    "- Only fix misspelled or non-existent words. Do not rephrase, add or remove words.\n"
    "- Do not change English words, names written in English, numbers or punctuation.\n"
    "- If nothing is wrong, answer [].\n"
    'Answer with JSON only: [{"wrong": "...", "correct": "..."}]'
)
P0_EXAMPLES = [
    ("เมื่อวานฉันไปซื้อผลไม้ที่ตลาดและได้ทุเรีนมาสองลูก", [{"wrong": "ทุเรีน", "correct": "ทุเรียน"}]),
    ("การประชุมครั้งนี้มีผู้เข้าร่วมจากหลายประเทศในเอเชีย", []),
    ("นักดาราศาสตร์ศึกษาการเคลื่อนที่ของดาวเคาระห์รอบดวงอาทิตย์", [{"wrong": "ดาวเคาระห์", "correct": "ดาวเคราะห์"}]),
]

P1 = """You are a careful proofreader for a Thai speech-recognition (ASR) system.

Context:
- The input is the automatic transcript of ONE sentence of Thai read speech, mostly news and encyclopedia-style text. A speech model wrote it while listening to the audio. You cannot hear the audio.
- The speech model is good: almost every word is already correct, and most sentences need no change at all.
- Its typical mistake is a word that SOUNDS right but is WRITTEN wrong: a wrong vowel or vowel length, a wrong tone mark, a similar-sounding but wrong consonant, or one missing or extra character. The result is usually a string that is not a real Thai word.

Task: find only those misspelled words and give the correct spelling of the word that was spoken.

Strict rules:
1. The correction must sound the same (or almost the same) as the wrong text. Never replace a word with a different word, a synonym, or a word that "fits better".
2. If the text is a real Thai word, leave it alone, even if it looks unusual in context. Sport terms, technical terms and Thai spellings of foreign words are often correct.
3. Never add words, remove words, complete a phrase, or rephrase.
4. Never touch English (Latin-letter) words, numbers, punctuation or spaces. Never translate English into Thai or Thai into English.
5. Foreign names written in Thai: fix one only when a single character is clearly wrong. Never invent a different name.
6. "wrong" must be copied exactly, character for character, from the input sentence.
7. When unsure, do not fix. The answer [] is usually correct.

Output JSON only, no explanation:
[{"wrong": "<exact text from the input>", "correct": "<correct spelling>"}]
or [] when nothing needs fixing."""
# Invented news-style sentences (not from train, dev or test). build.py checks that none of the
# example "wrong" strings occur in the real inputs, so an echoed example can't change anything.
P1_EXAMPLES = [
    ("นายกรัฐมนตรีเดินทางไปเยือนประเทศญี่ปุ่นเพื่อหารือเรื่องความร่วมมือทางเศรษกิจ",
     [{"wrong": "เศรษกิจ", "correct": "เศรษฐกิจ"}]),
    ("ทีมชาติออสเตรเลียทำได้ 250 รันก่อนที่ฝนจะตกจนต้องหยุดการแข่งขัน", []),
    ("แผ่นดินไหวครั้งนี้ทำให้อาคารหลายหลังพังถล่มและมีผู้บาดเจ้บจำนวนมาก",
     [{"wrong": "บาดเจ้บ", "correct": "บาดเจ็บ"}]),
    ("การประชุม G20 ครั้งนี้จัดขึ้นที่กรุงโรม โดยมีผู้นำจากหลายประเทศเข้าร่วม", []),
    ("ตำรวจกล่าวว่าผู้ต้องสงสัยถูกจับกุมได้ที่สถานีรถไฟในเมืองแมนเชสเตอร์เมื่อคืนวันอังคาร", []),
    ("อุณหภูมิของมหาสมุทรที่สูงขึ้นทำให้ปะการังฟอกขาวเป็นบริเวรกว้าง",
     [{"wrong": "บริเวร", "correct": "บริเวณ"}]),
    ("นักวิจัยใช้กล้องโทรทรรศน์อวกาศเจมส์ เว็บบ์ ศึกษาดาวฤกษ์ที่อยู่ห่างออกไปหลายพันปีแสง", []),
]

RUNS = [
    ("4b_P0", "typhoon-ai/typhoon2.1-gemma3-4b", False, P0, P0_EXAMPLES),
    ("4b_P1", "typhoon-ai/typhoon2.1-gemma3-4b", False, P1, P1_EXAMPLES),
]


def worker(run_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name, model_id, four_bit, system, examples = next(r for r in RUNS if r[0] == run_name)
    kw = {"device_map": "cuda", "torch_dtype": torch.bfloat16}  # Gemma 3 overflows in float16
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    print(name, "loaded", model_id, flush=True)

    shots = []
    for text, fixes in examples:
        shots += [{"role": "user", "content": text},
                  {"role": "assistant", "content": json.dumps(fixes, ensure_ascii=False)}]

    path = os.path.join(OUT, f"replies_{name}.json")
    replies = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    todo = [(k, s) for k, s in DATA["sentences"].items() if k not in replies]
    t0 = time.time()
    for i, (key, sentence) in enumerate(todo):
        msgs = [{"role": "system", "content": system}] + shots + [{"role": "user", "content": sentence}]
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True).to(llm.device)
        with torch.no_grad():
            out = llm.generate(**inputs, max_new_tokens=160, do_sample=False)
        replies[key] = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            json.dump(replies, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{name}: {i + 1}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        return
    os.makedirs(OUT, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"])
    print(len(DATA["sentences"]), "unique sentences", flush=True)
    for run in RUNS:  # each run in its own process, so a failure (e.g. 12B out of memory) can't stop the others
        done = subprocess.run([sys.executable, __file__, "--worker", run[0]])
        print(run[0], "finished" if done.returncode == 0 else f"FAILED (exit {done.returncode})", flush=True)


if __name__ == "__main__":
    main()
