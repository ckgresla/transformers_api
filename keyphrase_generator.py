# CKG Module for BRIO Summarization, alt to GPT-3 Generated (cheaper)

import os
import pickle #optional for saving binarized models
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration #summarizing module
from transformers import AutoTokenizer, AutoModel
from transformers import TokenClassificationPipeline, AutoModelForTokenClassification

from transformers import logging


# Keyphrase Generator (wraps base HF Class)
class KeyphraseExtractor(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model=AutoModelForTokenClassification.from_pretrained(model), tokenizer=AutoTokenizer.from_pretrained(model, model_max_length=512), *args, **kwargs)
        # Instantiate Model & Params
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "ml6team/keyphrase-extraction-kbir-inspec" #name as per hf download– https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec
        # Original Paper– "https://arxiv.org/pdf/2112.08547.pdf"
        self.name = 'KBIR-Inspec' #natural language name for saving (user set)
        self.max_length = 512 #as specified in HF Docs
        self.idx2label = {0: "B", 1: "I", 2: "O"} #mapping for classes in model generation


        logging.set_verbosity_warning() #remove warning, not training
        logging.set_verbosity_error() #remove warning, not training

        # Load in Pre-Trained Model & Tokenizer (can save artifacts to disk as if it speeds up load-in/inference)
        if os.path.isfile(f"./models/{self.name}-Tokenizer.pt"):
            file_to_unpickle = open(f"./models/{self.name}-Tokenizer.pt", "rb")
            self.tokenizer = pickle.load(file_to_unpickle)
            print(f"Loaded Tokenizer for {self.name} from Disk")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"Loading in Tokenizer for {self.name} from HF")

        if os.path.isfile(f"./models/{self.name}-Model.pt"):
            file_to_unpickle = open(f"./models/{self.name}-Model.pt", "rb")
            self.model = pickle.load(file_to_unpickle)
            print(f"Loaded Model- {self.name} from Disk")
        else: 
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            print(f"Loading in Model- {self.name} from HF")

    # Util to Get Output from Model into Atomic Keywords (useful format)
    def parse_keywords(self, keyphrases):
        relevant_tokens = [] #to hold the relevant spans (after parsing output)
        keyphrases = [(i["word"], i["entity"][0], i["start"], i["end"]) for i in keyphrases] #get relevant portions for output

        for word, label, start, end in keyphrases:
            word = word.strip("Ġ").strip()
            if label == "B":
                relevant_tokens.append([word])
                prev_end = end #for checking next span
            elif label == "I":
                if len(relevant_tokens) > 0:
                    # Get Consecutive Spans together (changes outputs from [key, phrase, extractor] into [keyphrase, extractor])
                    if prev_end == start:
                        relevant_tokens[len(relevant_tokens) - 1][-1] += word #if current KeyPhrase  span is part of last word
                        prev_end = end #make sure all single word spans are in single word, not spread out as sep tokens
                    else:
                        relevant_tokens[len(relevant_tokens) - 1].append(word) #appends new word to last list of KeyPhrase Spans
                        prev_end = end #combine all sep token spans into proper words

        # Returned Formatted Output (list of KeyPhrases, not list of lists of relevant tokens)
        relevant_tokens = [" ".join(i) for i in relevant_tokens]
        return relevant_tokens


# Save Weights for Embedding Model
extractor = KeyphraseExtractor(model="ml6team/keyphrase-extraction-kbir-inspec")

# Save Model Artifacts to Dir
with open(f"./models/{extractor.name}-Tokenizer.pt", 'wb') as fh:
    pickle.dump(extractor.tokenizer, fh)

with open(f"./models/{extractor.name}-Model.pt", 'wb') as fh:
    pickle.dump(extractor.model, fh)

