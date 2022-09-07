#!/home/ckg/miniconda3/envs/brio/bin/python3
# Test the Comparison of Actual Article Sentences against Summary Points (Explainability of Summary | get to highlight relevant original text)

import torch
from transformers import BartTokenizer, BartForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration
from transformers import logging


# File Names to Pick: ["bancor", "higgsboson", "kNNwiki", "recipeblock", "superres"]
file_name = "recipeblock"
article = open(f"./sample_text/{file_name}.txt").readlines()
article = [i for i in article if i!="\n"] #remove all non-text lines
article = " ".join(article) #convert list of article sentences into single sentence (full-article gets abstract summarization)

IS_CNNDM = True #false uses XSUM dataset & Pegasus Model (Pegasus trained for GAP Sentence Pred as opposed to CNN-DM BRIO) 
                #anecdotally, XSUM returns shorter summaries and CNN-DM returns multi points
max_length = 1024 if IS_CNNDM else 512

logging.set_verbosity_warning() #remove annoying warning, not training here
logging.set_verbosity_error() #remove really annoying warning

# Load in Pre-Trained Model (both set to Case Variant -- i.e different case words have different tokens)
if IS_CNNDM:
    # BART Pre-Trained
    model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
    tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
else:
    # Pegasus Pre-Trained
    model = PegasusForConditionalGeneration.from_pretrained('Yale-LILY/brio-xsum-cased')
    tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Generate Summary
def summarize(article):
    inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
    inputs.to(device) #gofast
    model.to(device) #gofast
    summary_ids = model.generate(inputs["input_ids"])
    output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

# Format Summary Output
def bpf(txt):
    bps = []
    for t in txt.split(". "):
        t = t.strip(".").title() #nice casing for Bullet Points
        bp = f"- {t}"
        # print(bp)
        bps.append(bp)
    return bps


# Generate Summary for Specified Article
output = summarize(article)
kps = bpf(output)

for i in kps:
    print(i)

# print(article) #used for debugging


