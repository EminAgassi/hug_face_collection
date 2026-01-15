#import whisper
#import subprocess

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

import time # For timing the generation

import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"]   = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())

print("ptxas:", shutil.which("ptxas"))


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

#transcript = process_speech()
#transcript = " Good morning, Mr. Thompson and Dr. Reynolds. How are you today? Good morning, Doctor. I've been better, honestly. My legs have been giving me trouble. I see from your chart you've been experiencing some discomfort. Could you describe your symptoms for me? Sure. For the past few months, I've had increasing pain and heaviness in my legs, especially after standing or walking for a while. Sometimes I'm swelling around my ankles too. I've just stood. Do you notice any changes in your symptoms when you elevate your legs? Yes, actually. The pain seems to ease quite a bit when I put my feet up. Have you noticed any visible changes, like vericose veins or skin discoloration? Yes, there are some dark, bluish friends on my calves. And my skin sometimes looks reddish or darker around my ankles. Have you ever had any ulcers or sores that don't heal well on your legs? No, thankfully nothing like that yet. Good, that's important to monitor. Do you have a history of blood clots or vein issues in your family? My mother have vericose veins, and I think my grandfather had some blood clot problems. How about your own medical history? Any previous surgeries, smoking habits, or other conditions? I used to smoke, but I quit about five years ago. I've had no surgeries, but I do have high blood pressure, diabetes, and arthritis. Are you currently taking any medications? Yes, I'm on list and apple for high blood pressure, met from an anglipid side for diabetes, and I take my legs sick and for arthritis pain. My blood sugar has been mostly under control, but I've noticed more joints diffusely. Thank you. And do you have any known allergies to medications or anything else? Yes, I'm allergic to pencil and it gave me a bad rash years ago. I also react to suffer drugs, so I avoid those two. That's good to know. We'll make sure to avoid those in any future treatments. Have you ever had any procedures or treatments done previously related to your veins or arteries? No, nothing related to that. All right, Mr. Thompson, let's go ahead and do a quick physical exam. Please have a seat on the exam table and swing your legs up so they're fully supported. I need to take a close look at both legs and check your circulation. Okay, no problem. I'm examining the skin on your lower legs for any signs of discoloration, swelling, or skin breakdown. I can see some mild hyperpigmentation around the ankles and some varicostities. Now I'm popating for pulses starting with the dosalis, pettis, and posterior tibulaturis. Both pulses are present and symmetric. I'm also checking for any signs of pitting edema pressing gently above your ankles. There's mild indentation that results quickly. I'm finishing with a quick check for any tenderness or temperature differences, nothing abnormal noted. Thank you for cooperating. You can sit back now. Your pulses are good, but I do notice signs consistent with venous insufficiency varicose veins and some mild skin changes. I'd like to order an ultrasound to assess your veins function. What will the ultrasound show? It will show how well your veins and valves function, and whether there's any significant reflux or blockage causing your symptoms. And what's the treatment if there is a problem? Treatments range from lifestyle changes, like using compression stockings and leg elevation, to minimally invasive procedures if the ultrasound shows significant reflux or varicose veins. In the meantime, I'd like you to try wearing compression stockings during the day, especially while you're working. They can really help reduce swelling and discomfort. We'll see how much relief you get from that by the time we meet again. Okay, that sounds manageable. When should I schedule the ultrasound? Let's have you get that done this week and we can review the results together next week. That sounds good. Thank you, doctor. You're welcome, Mr. Thompson. We'll get you feeling better soon. Take care."

# Remove problematic characters
#clean_transcript = transcript.encode("utf-8", errors="ignore").decode("utf-8")


from huggingface_hub import login
import os
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)  # programmatic login
else:
    # Token not found, login will not be performed
    pass

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_id = "google/gemma-3-4b-it"
#model_id = "meta-llama/Llama-3.3-70B-Instruct"
#model_id = "meta-llama/Llama-3.1-8B-Instruct"

device   = "cuda" if torch.cuda.is_available() else "cpu"
dtype    = torch.float16 if device == "cuda" else torch.float32

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model.
# Using 'Qwen/Qwen2-7B-Instruct' as an example.
model_name = model_id
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fix: Set pad_token to eos_token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#Check for bfloat16 support
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    print("GPU supports bfloat16")
    dtype = torch.bfloat16
else:
    print("GPU may not support bfloat16, falling back to float16 with AMP")
    dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype, #torch.float16, # Use float16 for efficiency
    device_map="auto", # Automatically load the model on the fastest available device
    attn_implementation="sdpa" # Enable Flash Attention 2
)

# Compile the model after loading it
model = torch.compile(model)

print(model.config._attn_implementation)

# Warm-up the model to avoid initial latency
print("Running warm-up generation...")
warmup_input = tokenizer("warmup", return_tensors="pt").to(model.device)
_ = model.generate(**warmup_input, max_new_tokens=5, use_cache=True)
print("Warm-up complete.")


#transcript = "Good morning, Mr. Thompson and Dr. Reynolds. How are you today? Good morning, Doctor. I've been better, honestly. My legs have been giving me trouble. I see from your chart you've been experiencing some discomfort. Could you describe your symptoms for me? Sure. For the past few months, I've had increasing pain and heaviness in my legs, especially after standing or walking for a while. Sometimes I'm swelling around my ankles too. I've just stood. Do you notice any changes in your symptoms when you elevate your legs? Yes, actually. The pain seems to ease quite a bit when I put my feet up. Have you noticed any visible changes, like vericose veins or skin discoloration? Yes, there are some dark, bluish friends on my calves. And my skin sometimes looks reddish or darker around my ankles. Have you ever had any ulcers or sores that don't heal well on your legs? No, thankfully nothing like that yet. Good, that's important to monitor. Do you have a history of blood clots or vein issues in your family? My mother have vericose veins, and I think my grandfather had some blood clot problems. How about your own medical history? Any previous surgeries, smoking habits, or other conditions? I used to smoke, but I quit about five years ago. I've had no surgeries, but I do have high blood pressure, diabetes, and arthritis. Are you currently taking any medications? Yes, I'm on list and apple for high blood pressure, met from an anglipid side for diabetes, and I take my legs sick and for arthritis pain. My blood sugar has been mostly under control, but I've noticed more joints diffusely. Thank you. And do you have any known allergies to medications or anything else? Yes, I'm allergic to pencil and it gave me a bad rash years ago. I also react to suffer drugs, so I avoid those two. That's good to know. We'll make sure to avoid those in any future treatments. Have you ever had any procedures or treatments done previously related to your veins or arteries? No, nothing related to that. All right, Mr. Thompson, let's go ahead and do a quick physical exam. Please have a seat on the exam table and swing your legs up so they're fully supported. I need to take a close look at both legs and check your circulation. Okay, no problem. I'm examining the skin on your lower legs for any signs of discoloration, swelling, or skin breakdown. I can see some mild hyperpigmentation around the ankles and some varicostities. Now I'm popating for pulses starting with the dosalis, pettis, and posterior tibulaturis. Both pulses are present and symmetric. I'm also checking for any signs of pitting edema pressing gently above your ankles. There's mild indentation that results quickly. I'm finishing with a quick check for any tenderness or temperature differences, nothing abnormal noted. Thank you for cooperating. You can sit back now. Your pulses are good, but I do notice signs consistent with venous insufficiency varicose veins and some mild skin changes. I'd like to order an ultrasound to assess your veins function. What will the ultrasound show? It will show how well your veins and valves function, and whether there's any significant reflux or blockage causing your symptoms. And what's the treatment if there is a problem? Treatments range from lifestyle changes, like using compression stockings and leg elevation, to minimally invasive procedures if the ultrasound shows significant reflux or varicose veins. In the meantime, I'd like you to try wearing compression stockings during the day, especially while you're working. They can really help reduce swelling and discomfort. We'll see how much relief you get from that by the time we meet again. Okay, that sounds manageable. When should I schedule the ultrasound? Let's have you get that done this week and we can review the results together next week. That sounds good. Thank you, doctor. You're welcome, Mr. Thompson. We'll get you feeling better soon. Take care."
#transcript = "Good morning, Mr. Thompson, and Dr. Reynolds. How are you today?  Good morning, Dr. I've been better honestly. My legs have been giving me trouble.  I see from your chart you've been experiencing some discomfort. Could you describe your symptoms for me?  Sure. For the past few months I've had increasing pain and heaviness in my legs,  especially after standing or walking for a while. Sometimes they're swelling around my ankles too.  Understood. Do you notice any changes in your symptoms when you elevate your legs?  Yes, actually. The pain seems to use quite a bit when I put my feet up.  Have you noticed any visible changes, like rare coast veins or skin discoloration?  Yes, there are some dark bluish rains on my calves and my skin sometimes looks  reddish or darker around my ankles. Have you ever had any ulcers or sores that  don't heal well on your legs? No, thankfully, nothing like that yet.  Good, that's important to monitor. Do you have a history of blood clots or vein issues in your  family? My mother had varicose veins and I think my grandfather had some blood clot problems.  How about your own medical history? Any previous surgeries, smoking habits or other conditions?  I used to smoke, but I quit about five years ago. I've had no surgeries,  but I do have high blood pressure, diabetes and arthritis.  Are you currently taking any medications?  Yes, I'm on this novel for high blood pressure, met from in and gliperside for diabetes,  and I take melexicum for arthritis pain. My blood sugar has been mostly under control,  but I've been noticed more joints stiffness lately.  Thank you. And do you have any known allergies to medications or anything else?  Yes, I'm allergic to penicillin. It gave me about a rash years ago.  I also react to sulphur drugs, so I avoid those, too.  That's good to know. We'll make sure to avoid those in any future treatments.  Have you ever had any procedures or treatments done previously related to your veins or arteries?  No, nothing related to that. Have you received any vaccinations recently or in the past year?  Yes, I got the flu shot a couple of months ago and my COVID-19 booster last fall.  I also got the shingles vaccine last year on my doctor's recommendation.  Excellent. It's good that you're staying up to date with your vaccinations.  Or write Mr. Thompson, let us go ahead and do a quick physical exam.  Please have a seat on the exam table and swing your legs up so they're fully supported.  I all need to take a close look at both legs in check your circulation.  Okay, no problem.  I am examining the skin on your lower legs for any signs of discoloration, swelling,  or skin breakdown. I can see some mild hyperpigmentation around the ankles and some very costus.  Now I am puppeting for pulses starting with the dosalis, pettis, and posterior tubular arteries.  Those pulses are present in symmetric. I am also checking for any signs of  pitting edema pressing gently above your ankles. They're s-mild indentation that resolves quickly.  I am finishing with a quick check for any tenderness or temperature differences nothing abnormal  noted. Thank you for cooperating. You can sit back now. Your pulses are good, but I do notice  signs consistent with thinness and sufficiency of our coast veins and some mild skin changes.  I'd like to order an ultrasound to assess your veins function.  What will the ultrasound show? It will show how while your veins  and valves function and whether there's any significant reflux or blockage causing your symptoms.  And what's the treatment if there is a problem?  Treatments range from lifestyle changes like using compression stockings and leg elevation  to minimally invasive procedures if the ultrasound shows significant reflux or varicose veins.  In the meantime, I'd like you to try wearing compression stockings during the day, especially  while you're working. They can really help reduce swelling and discomfort. We'll see how much  relief you get from that by the time we meet again. Okay, that sounds manageable. When should I  schedule the ultrasound? Let's have you get that done this week and we can review the results  together next week. That sounds good. Thank you doctor.  You're welcome, Mr. Thompson. We'll get you feeling better soon. Take care."
#transcript="I'm starting recording and like don't pay attention.  Try not to.  So imagine that you're subscribed.  Okay.  And a student writing down things that you want to ask and note.  That's cool.  Okay, go ahead.  So I'll just kind of go about how I normally would.  So, Natalie, what is your date for?  10, 14, 17.  All right.  Do you know your current weight?  Yes.  It's 172 pounds.  Okay.  Very good.  All right.  So you were seen in our office last and October of 2024.  You had a vein ablation done?  Correct.  Okay.  Both legs.  All right.  So what brings you in to see Dr. Gersoffi today?  So now, well originally my pain has been around the cough and the sheen.  And now it's more migrated to ankle above the bone and below.  And sometimes I feel my heal is in pain.  Also, I feel that my legs are swollen and it's very interesting sensation because at night,  well, when I wake up, I don't feel rested.  It's okay.  But like putting legs up gives a little bit of relief, but not 100%.  Okay.  And sometimes I feel like my skin at the ankle area, like about to burst.  Like it feels like it's like very, like, right.  Exactly.  That's how I feel.  But at the same time, well, I see that there is inflammation so it feels too tight.  Okay.  So it started, I would say, about three months ago, not right away.  Okay.  All right.  So you had some results from the procedure, but now, three months, it started bothering again.  Right.  In the slight of different place.  Okay.  And it started at the calf and now it's proceeded down to your ankle.  Yes.  Okay.  All right.  Okay.  All right.  Okay.  All right.  Okay.  All right.  All right.  Okay.  I'm just going to get your full of pressure.  Okay.  Then I'll call it.  As you know, even now I take all the info.  He's going to ask you, you know, where we're going to go over top.  Okay.  Okay, well, it's nice and slippery, take a tie there.  Okay, there we go.  Are you still wearing your compression stockings?  Good.  Okay.  Okay.  1520.  Okay.  Let me not tie these ones.  Should I push her on the lower side?  Well, typically.  Typically it's like one 10 or over 70.  What are you doing here?  I got one 12 or 70.  Oh, that's perfect.  Good stuff.  Just curious.  All right.  There you go.  I'll let him do your thing.  Thank you."
#transcript="all right so it's just a quarter to four nothing else hello so what brings you in today  well I have pain around above my ankle and below my ankle  as well as a little bit behind my foot my foot right it's different from what I had before before I  had mostly pain on my sheen and my cup I'm making sure I'm doing not but now it's actually  my greatest so and another issue is that so in the evening I feel very tired it feels like  my skin wants to burst at the bottom of my feet and I was thinking maybe it's a dry skin so I  moisturize it and help and when I go to sleep it's hard to go to sleep without a proffin and then  in the morning I don't feel that I got enough rest my feet still sore so then and then  what else is interesting so I guess everything is around this hill area below and above a little bit  so before I had it more on the higher level at the sheen level but now  is there any time of day where you would say if the pain is better or worse in the morning is  better but not fully I don't feel fully recovered okay it's just some tingling some pain but not a lot  at night I feel I feel like I want to put my leg somewhere especially during day I also try to  work my leg wear stockings do you see any improvement with the use of stockings there is some but  I still don't feel relieved in the morning so my sort of indication is that well during the night  it's enough time to keep it up the legs but elevated but in the morning I still not fully recovered  even though there is no real pain it's more like a discomfort do you have pain when you bear  weight on the ankle or on the foot even more and if you elevate your legs does that helps it helps  with time it helps so I need to be in elevated position for about like 10 minutes and then  yes okay have you had any injuries to the area at all no okay great all right well that sounds  pretty straightforward if you run from head to toe any headaches blurry vision chest pain shortness  breath not really I don't really exercise but one day I need to run to a bus with my boys and I  feel I'm short of breath feel scared me okay so that's just one time but I don't try it again so  when I isolated isolated it's a bit right but nothing else special no headaches and blood pressure seems  normal when when you lay down and sleep at night no difficulty breathing you're nothing I will go  okay all right well let's take a look at your leg and see what that skin looks like  I'm going to check your feet for pulses  you have very strong pulses so that's great  so this is the pain located this area here my ankle does it hurt if I touch yes just look here  here and here like close up to the bone I'm going to just passively flex and extend does that  make things better or worse either just a little bit any difference actually when you  lift it up it flex it it's better a little bit  and I'm just going to gently turn your foot one direction on the other  okay good this no no difference  all right your skin's healthy there's no rashes or ulcers so that's great we'll check the other side too  I'm pretty symmetrical it's on both sides or is it just the I think  all right a little bit worse okay  so this is the area yeah close it I see close it okay  and tender to touch as well  and the skin's healthy no rashes or ulcers there's no significant swelling there there may be  mild swelling but nothing with looks terrible wrong  okay  there's some kalangiotasia on the left  okay well my thoughts would be  venous insufficiency could potentially explain some of it  um the fact that there's point tenderness at the ankle would also suggest that there's a musculoskeletal  component to the symptoms so the musculoskeletal component um it could be a hairline fracture  it's unlikely in both legs so I would really put that lower on the differential diagnosis  um the ankle joint itself bears a lot of weight there's a lot of mechanical stresses that are  placed on it generally those types of complaints improve with resting the use of support stockings  not only supports your veins but it will also support the joint so it would take some of the  pressure off the joint and I wouldn't be surprised if that actually helps the ankles to feel  better um but basically resting I think would be helpful ultrasound testing would be helpful to  evaluate the extent of the venous insufficiency if it's a contributing factor or not and we can  use the compression stockings as a bit of a litmus test so if your symptoms improve significantly  that would be reassuring it would help us figure out that that were on the right track and if  you don't really see any improvement with the compression stockings that would probably take us  in a different direction to suggest more of a musculoskeletal component over the the circulatory  component should I try to wear stockings at night too um you can wear them really up until you get  into bed um once you're in bed and you lay flat the support that the stockings provide is really  minimal um so so there's no point to sleep in them for sure and uh it's fairly customary for  people to take them off as they're as they're getting into bed okay so what else do you document  typically yeah so the the review of systems um huh you know if there's any GI complaints um  if there's uh back pain leg pain changes to the skin um which appear to be negative um  you know in terms of our exam you are generally well appearing uh your your  symmetric um sweating uh your calm and appropriate um you breathe it in normal pace your  pulse rate is normal um you're not in any discomfort uh your joints move  uh symmetrically uh there's only point tendermis at the ankle areas  and in terms of your abdomen it's it's flat there's no discomfort with movement  so I think um I think overall the pretty focal findings here to suggest either musculoskeletal  issues at the ankles or venous insufficiency um we use a seep score um you know with trace  of damage your seep score would be three um your venous clinical severity score in the absence of  skin changes uh it would not be uh drastically high score um so in terms of our planning you know  for chronic venous insufficiency we'll use compression stockings we'll check a venous  reflux test and um we'll see you in follow up after the testing in terms of the ankle discomfort  where it may be related to musculoskeletal problems um if there are continued problems and there's  no improvement with the compression then the x-rays may be helpful as a as a workup of the musculoskeletal  component um and then we can take it from there sounds good thank you so when does my next follow-up  appointment we can do it after the ultrasound okay so the next what should I do next that's for sure  so um use the stockings regularly um the ultrasound we'll have that scheduled in the timely manner  and once that's finished uh we can review the uh the results together um you know you've been  using compression stockings for many months so um to to use them from the beginning of the day until  the end of the day uh I think we will help to set shed some light on um whether or not they're  factors okay thank you so much you're welcome let's go"
#transcript ="Go ahead. High blueberry pie, what brings you in today?  Well, I have this vein, and it keeps bruising like spontaneously, and I'm not sure what's going on.  On the right leg there.  On the right leg, yeah.  It's just this side.  And when did it start giving you trouble?  Well, about three weeks ago, I noticed it.  I thought maybe I had had something injured, but I'm not sure.  Okay.  And had you noticed the vein before then, or was it the first time you saw it?  No, sometimes when I run, I notice they get pretty pronounced after I'm running.  But other than that, I don't know if maybe the vein broke from my running, or what I did.  Does it change during the day?  No.  Is there any pain associated with it?  Yes.  Okay.  What does it feel like?  What's the character?  It's kind of a, like a throbbing.  Okay.  And then do you get any swelling?  No, I didn't notice any swelling.  Anything that makes it feel better or worse?  No.  Not that I can think of.  Do you use compression stockings?  I do when I run.  Okay.  And how long have you used the stockings?  About five years.  Okay.  Do you have any headaches, vision changes, chest pain, shortness of breath?  I have headaches, but I've had headaches my entire life.  Okay.  So I haven't noticed an increase of headaches.  My mother had very bad varicose veins.  She never had anything done with them.  She didn't have them treated, but they were very pronounced.  Okay.  Any belly pain besides the bruising and any skin changes?  No.  Not that I can think of.  Okay.  Joint pain?  No.  Any numbness or unilateral weakness?  No.  Do you have any medical problems besides the vein issue that you're here for?  No.  Not that I can think of.  Okay.  I have trouble sleeping.  Any prior surgeries?  I had two of my toes removed on the other foot because of an accident.  Okay.  A hysterectomy, and I've had both hands, a carpal tunnel on both hands.  All right.  Do you drink smoke or do any drugs?  No.  And you said your mother had varicose veins?  Very bad varicose veins.  Okay.  Yeah.  There's no diabetes, nothing.  I'm healthy other than my vein.  Okay.  So on an exam, there's no acute distress.  The answer appropriately.  Your eyes are clear.  Your hearing is normal.  Your neck has no...  Distended jugular veins.  Your respiratory rate is normal.  Check your pulse here.  Parcrate is normal.  No pain in the tummy?  No.  Okay.  Traced the demon in the lower extremities.  There's a bruise on the right medial ankle.  There's a varicose vein that's tender to touch.  You're able to move your joints without discomfort.  Oh, I forgot to tell you.  I did have sclerodone 12 years ago on this upper portion here.  Okay.  The physician I worked for.  Is it for me?  Okay.  So it's procedure history of sclerotherapy on the right leg.  Got it?  All right.  Well, it looks like this is probably a progression of chronic venous insufficiency.  We recommend that you continue to use compression stockings.  We'll get an ultrasound to look at your veins to see if there's any clots or venous reflux,  which is what I suspect they'll probably find reflux.  That's valvular incompetence.  And depending on what we find, that can direct further treatment.  You can just...  Yeah.  So basically what the information provided, you know, one of the goals would be,  does it capture the C classification accurately?  Is it able to calculate a venous clinical severity score accurately?  And I would think we would be able to do that.  I mean, just depending on how it deals with the unknown variables.  You know, if it just defaults to like the lowest setting or...  So medication-wise right now I'm taking a medial dose pack. I'm taking a moxacillin because I just  have my tooth taken care of. I take melatonin at night to sleep. I take an acyclovir when needed  for a cold sore that I get through stress. And what else do I take? Oh, I take maloxicam, 7.5  milligrams daily. Other than that, I don't take anything else. Do you have any allergies? No.  I have allergies. No allergies, no medical conditions. I'm very healthy over them.  Any sleeping, I have trouble sleeping. Any important medical procedures that happen to me?  It's directing me and my hands and I lost my toes."
#transcript = "Hello Apple Pie, what brings you in today?  Well, I'm not entirely sure. My primary doctor said I should be seen  because I'm having a lot of pain when I walk.  I can go from like my kitchen into the living room  and my right leg just feels terrible. Like I can't lift it. It hurts.  Okay, do you have any problems in the left leg?  No, the left leg doesn't seem to be bothering me. Although sometimes it feels heavy  but it doesn't, I don't have the pain in the left leg that I do in the right.  Okay, when did you first notice that pain?  About three weeks ago.  Okay, and you had no pain prior to that?  I don't really, I wasn't paying attention to it.  Okay, and what is the character of the pain?  Is it a cramp? Is it a burning sensation? What do you experience?  It feels like almost like it goes numb and then it starts to throb.  Okay, and then what happens when you're rest?  It dissipates a little bit but as soon as I get up it starts to hurt again.  Okay, and is your walking distance consistent  where you're only able to get that distance?  Yes, every day.  Am I correct in that the pain replicates itself?  Yes, it stops me from doing my normal activities.  Okay, do you have any ulcerations on your feet?  No, but if you notice I've got this like reddened area here down by my ankle  and my toe here is starting to look like it's rubbing against my shoe  but it's not my shoe because I have a worn shoes all summer.  Okay.  Do you have any medical problems?  I'm diabetic, I take metformin and just metformin  and I try to control it well with diet and I do smoke.  Okay, how long have you been a smoker?  Since I was 16 and 56 now.  And how many cigarettes do you smoke a day?  Generally between a quarter to a half a pack a day.  Have you had any surgeries?  The only surgery I ever had was a hysterectomy.  Are you allergic to anything?  I'm allergic to a penicillin and bactram.  Okay, what happens when you take those medicines?  Penicillin, I'm not sure they told me that as a child,  but bactram I swell and get very patchy itchy skin.  And besides the metformin, do you take any other medicines?  I take medicine, I forgot all about it, I have high blood pressure  but it's controlled with the medicine, I take low sartan.  Okay.  And you said you smoke cigarettes, do you drink alcohol or do any drugs?  I drink whiskey if I'm out with my friends.  I don't do any drugs now, but I certainly did when I was a teenager.  And do you have any problems in the family?  Yeah, actually my dad was severe diabetic.  My dad did lose his right leg below the knee.  And if I remember, and I'm sorry that I don't,  but if I remember, I do kind of think someone said something about arterial disease  but he's gone now, so I don't know.  Okay.  And have you had any favors or chills lately?  Not that I think of.  No.  Any changes to your vision?  No.  Any problems with hearing?  Well that depends on who you ask.  No, I get headaches, but I've always gotten headaches.  Okay.  So your hearing's okay?  Yeah.  And your chronic problem for you?  It's just this weird.  And sometimes my skin turns like a dry patchy color.  And I don't really think that it swells, but it gets kind of reddish.  Like can you see here?  It looks a little darker here then.  So I'm not really sure.  Yeah, and that's done by your ankle.  And that's been there as long as I can remember,  but I never really took notice of the exact date when it came.  Any chest pain, shortness of breath, difficulty breathing?  No, I'm a smoker, so sometimes if I exert myself, it's hard.  Like going up a couple of flights of stairs.  I know that it's because I'm a smoker.  I get a little winded with the activity, so okay.  Any abdominal pain or change in your bowel habits?  No.  Any joint pain?  My fingers.  Yeah, my fingers get very sore.  Especially at a winter time.  And any unilateral weakness or sensory changes?  What would you mean by sensory changes?  I had mentioned you had some numbness in the right foot.  Anywhere else?  No, not that I couldn't think of.  So hopefully it pulls in to the rear systems.  Which she has.  So one example, while appearing, no acute distress.  Answer questions appropriately.  Your sclerot is clear.  Your hearing is normal.  Your neck shows no distended jugular veins.  Your breathing normally.  Your heart rate's regular.  You have the rash one here.  Or the redness on your ankle.  Red ankle and foot.  You have the menace.  Flat.  Passive range of motion in the joints is normal.  One example.  There's non-palpable fetal pulses.  In the feet.  So your symptoms sound consistent with arterial disease.  So we'll get an arterial duplex to evaluate the arteries  in your lower extremities.  We'll have you engage in a walking program to try to improve  your walking distance.  People with circulation problems are often able to improve.  They're walking distance with physical activity alone.  Also, the risk for limb loss is quite low with the types of symptoms that you have.  So that gives us the opportunity to explore non-invasive options.  First, smoking cessation is a priority.  People that continue to smoke with arterial disease,  they rarely will see improvements with activities.  And it adversely affects any potential treatments that you may have to receive.  We'll add an aspirin to your daily medical regimen.  Patients with arterial disease, if it's confirmed with duplex,  have an indication to begin statin therapy,  so that's something that we'll consider for you.  Is that when people take statins because they have cholesterol problems?  It's usually prescribed for cholesterol problems,  and for people that have symptomatic arterial disease,  that is a solid indication to begin statin therapy,  regardless of cholesterol number.  So we will get those tests for you,  and then we'll follow up and see how you're progressing.  Dr. Can I ask you a question?  My friend has arterial disease and uses a pump on her leg.  Is that something that would be necessary if my arterial disease was that bad?  Yeah, that's a non-invasive option that can potentially help improve your symptoms.  So I think it's a great idea,  and if you'd like to explore that,  we can certainly arrange for a pump.  That would be wonderful, I'm sure.  Well, I'm very glad I came here today.  Likewise.  All right, thank you so much."
transcript = "All right.  Go ahead.  Okay.  I got hold on, Flynn.  What brings you in today?  Well, my primary care doctor sent me here because I have this gaping wound on my foot.  And it started out, and obviously you can see it.  It started off very small.  I thought maybe it was a spider bite, but it got worse and worse and worse.  And now, you know, the primary was treating it, but look at it.  I mean, there's clearly something going on, and it's unbelievably painful, and I just  can't take this anymore.  So when did that one start?  I want to say, between three and four weeks ago, like I said, it started off as a, I thought  it was a spider bite, and it, it's swollen, it got swollen, it got red, and then the skin  broke.  And how would you characterize the pain associated with the ulcer?  Unbelievably painful.  It burns, it throbs, it, it aches all the time, whether I'm standing or lying down,  nothing I do gives me relief.  They try to ibuprofen, they try to antibiotics.  And before you develop the ulcer, did you have any problems at all with your legs or  walking?  You know, I didn't think so, but now that I think about it, my legs of late, maybe the  last couple of months were tired and fatigued feeling, and even if I lay down, I would say  to my husband, can my legs just hurt?  Okay.  Does it affect your walking distance?  It does, it does.  And I didn't realize it until my husband said, you know, what's going on with you?  He made a comment that I was out of shape, because I used to be able to walk a couple of  blocks with him to walk our dogs, but now I can't even make it around the block.  And when, when you rest, does that have any effect on the way that your legs feel?  I think it's a different feeling, but they still hurt.  They have that.  It's like a fatigued, squeezing, painful, like nothing I do takes it away.  I've tried Tylenol, I've tried Advil, I tried Muloxacam, I tried some muscle relaxers,  nothing seems to be working for me.  And then the wound showed up.  Do you have any medical problems like you know of?  I'm borderline diabetic, but my primary doesn't think this is from diabetes, so I've  got hypertension, I have cholesterol issues, my triglycerides are a problem, so I do take  medications for those.  Okay, have you ever had any surgeries?  No.  Are you allergic to anything?  I am allergic to levoquin.  Okay, what happens when you take levoquin?  I vomit.  Okay.  And what medicines do you take at home?  Right now I'm taking ibuprofen with extra strength Tylenol, I take a tour of a statin,  I take low sarten, and I'm not taking any medication for diabetes, it's controlled  with diet, because like I said, they said I'm borderline.  So I take paxil for anxiety, and I take melatonin to sleep.  I used to take a sleeping med, whose name I can't remember right now, but I don't take  that any longer, because I couldn't wake up.  Ambient.  I used to take it earlier.  Okay.  Do you drink smoke or do any drugs?  I smoke on occasion socially, I do, I drink whiskey socially, but no, I don't do any street  jokes.  Any illnesses in your family?  My father has arterial disease, my mother died of brain cancer, and my brother is diabetic,  and has coronary artery disease.  Have you experienced fevers or chills?  Yes.  Okay.  And how often are you experiencing this?  I noticed I just had like maybe last week or the week before, and that's why the doctor  put me on an antibiotic.  Okay.  And that was for the foot.  That was for the foot, and I think the fever and the chills came from the foot, but  then he said that he, you know, with the antibiotic, are you still getting fevers and chills?  No.  Okay.  No, but he did say that he thought it was really important that I come to see a vascular  surgeon.  Sure.  Any changes in your vision or hearing?  No, not that I can think of.  Any difficulty swallowing?  No.  Any abdominal pain?  No.  Chest pain or shortness of breath?  No.  Uh, joint pain?  Just around where the, you know, it hurts to bend my foot.  Okay.  And you have the ulceration on your foot.  Okay.  Um, it's on exam or, uh, well appearing, no acute distress, uh, answer questions normally.  Your hearing is normal, your, uh, scleris clear, um, your respiratory rates normal, uh,  your heart rate is regular, uh, your abdomen is flat and untender.  Uh, you're able to move your upper and lower extremities symmetrically with normal range  of motion.  Uh, there is a ulcer on your left plantar four foot and it's one by two centimeters, uh,  where, uh, there's peripheral, erythema, suggestive of cellulitis, there's non-palpable  fetal pulses, um, so, um, our impression here is a, um, uh, non-pressure ulcer on the left, uh, four foot, um, the subtenus fat layer, uh, appears to be due to type two diabetes  and, um, arterial disease, um, associated cellulitis, um, I'm going to recommend continued  antibiotics, uh, with close surveillance, uh to, to ensure there's no progression of  the cellulitis, um, for the arterial disease, we're going to do arterial duplex testing  to establish a, uh, a baseline and if there's, uh, significant disease that will likely  require, uh, revascularization, uh, for the type two diabetes, malitis, which I can  even love at anyone's sea level, um, to ensure you're well controlled with your baseline,  um, and, uh, you know, strict avoidance of, uh, smoking, um, will add a daily aspirin  to your medical regimen, and, um, I'd like to see you back, um, within five days to  reassess the foot, if there's any, uh, progression of the redness or if there's increased pain,  uh, if you develop fevers or chills, again, um, you know, go immediately to the emergency  room and, uh, and that could potentially be, um, an ascending infection on that as a significant  risk for lung loss, with the, um, ulcer on your foot. Um, if we find arterial disease,  uh, we really have to intervene on that because, uh, the risk of major amputation without  revascularization is an excess of 90 percent at five years. 90 percent. Yeah. So, uh, it's  a serious problem. Is this because I was smoking doctor? It's smoking contributes to it,  so my head's genetics and, you know, I wish we, so no more stuff. All the variables for  smoking and essential little effect outcomes. Um, is there anything I should do with it?  Should I keep it wrapped? Should I keep it open? Uh, I keep pressure off that area. I'll  dress it with, uh, top glenobiotic and, um, go as rap, and, um, you have to, uh, avoid pressure  and keep the area dry. Oh, so no showering. Okay. All right, hopefully we can get the ultrasound  soon and get some answers. Thank you so much. You're welcome."
questions = load_questions("questions.json")
# Join questions in numbered list
q_block = "\n".join(questions)

prompt = f"""
Your persona is a medical scribe.

The transcript is produced by Whisper Local. Please be aware that this ASR system may have some inaccuracies.
{transcript}

Answer these questions:
{q_block}
"""

# Prepare the model input.
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
print(prompt)

print("\n--- Start Chatbot ---")
print("Type 'quit' or 'exit' to end the conversation.")

while True:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, return_attention_mask=True).to(model.device)

    # Generate the response
    start_time = time.time()
    
    # Generate text.
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=1500, # Adjust the maximum length of the generated response
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    end_time = time.time()

    generated_ids = [
        output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Print the generated response.
    print(response)
    print(f"(Generation took {end_time - start_time:.2f} seconds)")

    # 4. Add the model's response to the history. This is the first 'model' turn.
    messages.append({"role": "model", "content": response})

    # Get user input
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    # Add user's new question to the history
    messages.append({"role": "user", "content": user_input})

print("--- End Chatbot ---")