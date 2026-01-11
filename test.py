
import whisper
import subprocess

import requests
import json

from typing import List, Optional

import os, shutil

# disable compile/inductor everywhere
#os.environ["TORCHINDUCTOR_DISABLE"] = "1"
#os.environ["TORCHDYNAMO_DISABLE"]   = "1"  # disables torch.compile

# avoid SDPA kernels that may try fancy backends
#os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
#os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"]   = "1"


import torch, triton
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())
print("triton:", triton.__version__)
print("ptxas:", shutil.which("ptxas"))

from transformers import AutoTokenizer, AutoModelForCausalLM


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

def build_prompt(transcript, questions):
    # Join questions in numbered list
    q_block = "\n".join(questions)
    return f"""Transcript:
{transcript}

Questions:
{q_block}
"""

def query_llama(prompt):
    '''
    result = subprocess.run(
        #["ollama", "run", "llama3", prompt],
        ["ollama", "run", "gemma3:4b"],
        input=prompt,
        capture_output=True, text=True
    )
    #print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    result = result.stdout
    '''
    # Build the request payload
    payload = {
        "model": "gemma3:4b",          # or llama3.1, mistral, etc.
        "prompt": prompt,
        "options": {
            "temperature": 0.8,        # 0.0 = deterministic, higher = more creative
            "top_p": 0.9,
            "num_predict": 1024
        }
    }
    resp = requests.post("http://localhost:11434/api/generate", json=payload, stream=True, timeout=600)
    # Collect streamed response
    output = []
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                output.append(data["response"])
            if data.get("done"):
                break

    resp = "".join(output).strip()

    return resp

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
        initial_prompt=None,
    )
    # first visit complete_visit_audio_20250726_132123
    # after results came back complete_visit_audio_20250727_164848
    result = model.transcribe("complete_visit_audio_20250726_132123.wav", **common)
    return result["text"]

transcript = process_speech()
#transcript = " Good morning, Mr. Thompson and Dr. Reynolds. How are you today? Good morning, Doctor. I've been better, honestly. My legs have been giving me trouble. I see from your chart you've been experiencing some discomfort. Could you describe your symptoms for me? Sure. For the past few months, I've had increasing pain and heaviness in my legs, especially after standing or walking for a while. Sometimes I'm swelling around my ankles too. I've just stood. Do you notice any changes in your symptoms when you elevate your legs? Yes, actually. The pain seems to ease quite a bit when I put my feet up. Have you noticed any visible changes, like vericose veins or skin discoloration? Yes, there are some dark, bluish friends on my calves. And my skin sometimes looks reddish or darker around my ankles. Have you ever had any ulcers or sores that don't heal well on your legs? No, thankfully nothing like that yet. Good, that's important to monitor. Do you have a history of blood clots or vein issues in your family? My mother have vericose veins, and I think my grandfather had some blood clot problems. How about your own medical history? Any previous surgeries, smoking habits, or other conditions? I used to smoke, but I quit about five years ago. I've had no surgeries, but I do have high blood pressure, diabetes, and arthritis. Are you currently taking any medications? Yes, I'm on list and apple for high blood pressure, met from an anglipid side for diabetes, and I take my legs sick and for arthritis pain. My blood sugar has been mostly under control, but I've noticed more joints diffusely. Thank you. And do you have any known allergies to medications or anything else? Yes, I'm allergic to pencil and it gave me a bad rash years ago. I also react to suffer drugs, so I avoid those two. That's good to know. We'll make sure to avoid those in any future treatments. Have you ever had any procedures or treatments done previously related to your veins or arteries? No, nothing related to that. All right, Mr. Thompson, let's go ahead and do a quick physical exam. Please have a seat on the exam table and swing your legs up so they're fully supported. I need to take a close look at both legs and check your circulation. Okay, no problem. I'm examining the skin on your lower legs for any signs of discoloration, swelling, or skin breakdown. I can see some mild hyperpigmentation around the ankles and some varicostities. Now I'm popating for pulses starting with the dosalis, pettis, and posterior tibulaturis. Both pulses are present and symmetric. I'm also checking for any signs of pitting edema pressing gently above your ankles. There's mild indentation that results quickly. I'm finishing with a quick check for any tenderness or temperature differences, nothing abnormal noted. Thank you for cooperating. You can sit back now. Your pulses are good, but I do notice signs consistent with venous insufficiency varicose veins and some mild skin changes. I'd like to order an ultrasound to assess your veins function. What will the ultrasound show? It will show how well your veins and valves function, and whether there's any significant reflux or blockage causing your symptoms. And what's the treatment if there is a problem? Treatments range from lifestyle changes, like using compression stockings and leg elevation, to minimally invasive procedures if the ultrasound shows significant reflux or varicose veins. In the meantime, I'd like you to try wearing compression stockings during the day, especially while you're working. They can really help reduce swelling and discomfort. We'll see how much relief you get from that by the time we meet again. Okay, that sounds manageable. When should I schedule the ultrasound? Let's have you get that done this week and we can review the results together next week. That sounds good. Thank you, doctor. You're welcome, Mr. Thompson. We'll get you feeling better soon. Take care."

# Remove problematic characters
clean_transcript = transcript.encode("utf-8", errors="ignore").decode("utf-8")

'''
prompt = f"""
You are a clinical documentation assistant.

Below is a transcript of a conversation between a doctor and a patient during a medical visit.

Please generate a structured SOAP note with the following sections:

1. **Reason For Appointment**: Patient-reported symptoms and concerns.
2. **Objective**: Observed findings, vitals, labs, and physical exam (if mentioned).
3. **Assessment**: Physician’s assessment, differential diagnosis, or working diagnosis.
4. **Plan**: Recommended next steps, medications, tests, or referrals.

Use clear medical language. If information is missing in any section, leave a placeholder like "[Not documented]".

Transcript:
{transcript}
"""
'''

prompt = f"""
Transcript:
{clean_transcript}

Answer these questions:
1. Reason For Appointment: Extract patient-reported symptoms and concerns only. One per line. No extra text. if none, output nothing.
2.1 Current Symptomps: List each item per line. No extra text. if none, output nothing.
2.2 Symptomps Location: 
2.3 Severity:
2.4 Activities of Daily Living:
2.5 Existing Management and Therapy:
2.6 Compression Hose:
2.7 Compression Hose Type: 
2.8 Compression Duration:
2.9 Previous Evalutation for Veins:
3.0 Prior DVT: 
3.1 Prior SVT: 
3.2 Prior Hemorrhage Requiring Phys. Care: 
3.3 Prior PE:
3.4 Prior Ulcer:
3.5 Prior Hypercoag States:
3.6 Prior Thermal Ablation Laser:
3.7 Prior Thermal Ablation RF:
3.8 Prior Nonthermal Ablation:
3.9 Prior Sclerotherapy:
4.0 Prior Vein stripping:
4.1 Prior Phlebectomy:
4.2 Prior Surface Laser:
6. Current Medications:
7. Past Medical History:
8. Family History:
9. Surgical History:
10. Social History:
11. OB History:
12. Hospitalization/Major Diagnostic Procedure:
13. Allergies:
14. Review Of Systems:
15. Vital Signs:
16. Examination:
17. Assessments: Physician’s assessment, differential diagnosis, or working diagnosis.
18. Treatment: Recommended next steps, medications, tests, procedures, or referrals.
19. Immunization: Any vaccines.
20. Preventive Medicine: Any counseling and preventitive screenings.
21. Visit Codes:
22. Follow Up: Time to next visit.
"""




#questions = load_questions("questions.json", ids=["6."])
questions = load_questions("questions.json")

system = "You are a clinical documentation extractor. Answer each question strictly based on transcript. No extra text. When listing medications, always replace any ASR variants with their normalized forms. Never output the ASR text, only the corrected medication names e.g. Listen-up is Lisinopril, metfromanglepizide is Metformin and Glipizide, licksickin is Meloxicam"

user = build_prompt(transcript, questions)

prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:"
print("ready to call model using " + prompt)

result = query_llama(prompt)

print("Done with the call.")

print(result)


'''

from huggingface_hub import login
import os
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)  # programmatic login
else:
    # Token not found, login will not be performed
    pass

tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
print("Tokenizer loaded OK")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda",   # auto-places on GPU if available
     attn_implementation="eager"
)

print("Model + tokenizer loaded OK")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me three fun facts about FHIR."}
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

# Gemma chat often ends a turn with <eot>; make both <eot> and <eos> stopping tokens.
eos_ids = []
if tok.eos_token_id is not None:
    eos_ids.append(tok.eos_token_id)
try:
    eot_id = tok.convert_tokens_to_ids("<eot>")
    if isinstance(eot_id, int) and eot_id != -1:
        eos_ids.append(eot_id)
except Exception:
    pass
# Fallback if nothing resolved:
if not eos_ids:
    eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []

pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or eos_ids[0])

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
         min_new_tokens=16,
        do_sample=True,                 # deterministic
        eos_token_id=eos_ids,            # <-- accept <eot> and <eos>
        pad_token_id=pad_id,
    )

# decode ONLY newly generated tokens
gen_only = outputs[0][inputs["input_ids"].shape[-1]:]
text = tok.decode(gen_only, skip_special_tokens=True).strip()
print("Assistant:", text if text else "[empty]")
print("new tokens:", int(gen_only.shape[-1]))
'''



