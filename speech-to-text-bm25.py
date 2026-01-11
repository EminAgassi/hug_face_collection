# fast whisper expirement

import whisper
from faster_whisper import WhisperModel
from rank_bm25 import BM25Okapi
import re
from typing import List, Optional
import json

import torch
import time

import os, shutil

import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"]   = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast

from minilm_extractor import E5LineExtractor

from llm_model import run_model 

def process_speech():

    model = whisper.load_model("base.en", device="cuda") # try "tiny" or "small" for faster performance

    common = dict(
        beam_size=5,               # beam search, no sampling
        temperature=0.0,
        best_of=None,
        condition_on_previous_text=False,
        fp16=True,                 # keep constant between runs
        word_timestamps=True,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        initial_prompt=None
    )
    # first visit complete_visit_audio_20250726_132123
    # after results came back complete_visit_audio_20250727_164848
    result = model.transcribe("complete_visit_audio_20250726_132123.wav", **common)
    return result

def search(question: str, k: int = 5, ctx: int = 0, rel=0.5, abs_min=1.2):
    q_tokens = [t for t in tok(question.lower()) if t not in STOP]
    if not q_tokens:
        return []  # nothing to match on

    scores = bm25.get_scores(q_tokens)
    # keep only nonzero scores
    cand = [(i, float(s)) for i, s in enumerate(scores) if s > 0.0]

    if not cand:
        return []  # no lexical match; avoid returning greetings

    # compute term overlap (at least 1 shared token)
    qset = set(q_tokens)
    def overlap(i):
        return len(qset.intersection(docs[i]))

    cand = [(i, s) for i, s in cand if overlap(i) >= 1]
    if not cand:
        return []

    # relative + absolute thresholding
    smax = max(s for _, s in cand)
    cand = [(i, s) for i, s in cand if s >= rel * smax and s >= abs_min]
    if not cand:
        return []

    # top-k after filtering
    cand.sort(key=lambda x: x[1], reverse=True)
    idxs = [i for i, _ in cand[:k]]

    # apply context window AFTER filtering so we don’t pull in greetings
    out = []
    for i in idxs:
        lo, hi = max(0, i - ctx), min(len(segs), i + ctx + 1)
        out.append({
            "hit_id": i,
            "start": segs[lo]["start"],
            "end": segs[hi-1]["end"],
            "text": " ".join(segs[j]["text"] for j in range(lo, hi)),
            "score": float(scores[i]),
            "overlap": overlap(i)
        })
    return out

def load_questions(path="questions.json", ids: Optional[List[str]] = None):
    """
    Load questions from JSON. If ids is provided, only return questions whose 
    number prefix (like '6.' or '2.1') matches.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]

    if ids:
        filtered = []
        for q in questions:
            for qid in ids:
                if q.strip().startswith(qid):   # e.g. "6." or "2.1"
                    filtered.append(q)
        return filtered
    
    return questions

def build_prompt(question, hits):
    
    # Concatenate only top segments
    
    context = "\n".join(f"[{h['time'][0]:.2f}-{h['time'][1]:.2f}] {h['text']}"
                        for h in hits['hits'])
    
    return f"""Your persona is a medical scribe.
Answer this question strictly from the transcript. If answer is not found, reply N/A.
Note, the transcript is produced by Whisper Local. Please be aware that this ASR system may have some inaccuracies. Please double-check all medication names.

Question:
{question}

Transcript:
{context}
"""

'''
result = process_speech()

# Segment timestamps
for seg in result["segments"]:
    print(f"[{seg['start']:.2f} --> {seg['end']:.2f}] {seg['text']}")

print(result["text"])
'''

# --- Model loading and compilation (happens once at startup) ---
# Check for bfloat16 support
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    dtype = torch.bfloat16
else:
    dtype = torch.float32

import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())

print("ptxas:", shutil.which("ptxas"))

# Load model with Flash Attention 2 enabled
model_name = "google/gemma-3-4b-it"
try:
    print("Loading model with Flash Attention 2...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
except ImportError:
    print("Flash Attention 2 not available. Falling back to standard attention.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = torch.compile(model)

# Warm-up pass to avoid initial latency
print("Running warm-up generation...")
warmup_input = tokenizer("warmup", return_tensors="pt").to(model.device)
_ = model.generate(**warmup_input, max_new_tokens=5, use_cache=True)
print("Model ready!")

wmodel = WhisperModel("base", device="cuda", compute_type="float16")

segments, info = wmodel.transcribe("complete_visit_audio_20250726_132123.wav", beam_size=6, word_timestamps=True)

'''
for seg in segments:
    print(f"[{seg.start:.2f} --> {seg.end:.2f}] {seg.text}")
'''

segs = [{"id": i, "start": float(s.start), "end": float(s.end),
         "text": s.text.strip()} for i, s in enumerate(segments)]

print(segs)

# --- 2) Build BM25 ---
tok = re.compile(r"[a-z0-9']+").findall
STOP = {"the","and","a","an","to","of","in","on","for","with","is","are","was","were"}

docs = [[t for t in tok(x["text"].lower()) if t not in STOP] for x in segs]
bm25 = BM25Okapi(docs)

extractor = E5LineExtractor()
extractor.fit(segs)

questions = load_questions("questions.json")

print("\n")

result = extractor.search("Current Medications: patient lists medicines they are taking for their conditions (e.g., blood pressure, diabetes, arthritis)", top_k=10, min_conf=0.30)
print(result)

'''
start_time = time.time()

#run_model(segs)

for q in questions:
    #hits = search(q, k=3, ctx=1)
    #hits = search(q, k=3, ctx=1)
    
    print(q)

    #prompt = build_prompt(q, hits)
    #print(prompt)

    result = extractor.search(q, top_k=5, min_conf=0.30)
    print(result)

    prompt = build_prompt(q, result)

    messages = [
    {
        "role": "user",
        "content": (
            prompt
            #"Your persona is a persian doctor. Only answer with persian-themed language. "
            #"What is the diabetes?"
        ),
    }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").to(model.device)

   
    with torch.no_grad():
        with autocast(dtype=dtype):
            generated_ids = model.generate(
                **input_ids,
                max_new_tokens=250,
                use_cache=True,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )
   

    # Decode and extract only the new response
    response_ids = generated_ids[0][len(input_ids.input_ids[0]):]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(" ----------------- Answer ----------------- ")
    print(response)
    

    print("\n")


end_time = time.time()
print(f"Generation took {end_time - start_time:.2f} seconds.")
'''

'''
print(f"\n=== {q} ===")
for h in hits:
    print(f"[{h['start']:.2f}-{h['end']:.2f}] score={h['score']:.3f}\n{h['text']}\n")

'''
    
    