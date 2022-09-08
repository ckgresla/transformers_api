# CKG Module for HF Model Spin Up

import os
import pickle #for saving binarized models
import torch
import transformers #for checking the version being used
from transformers import BartTokenizer, BartForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration #summarizing module
from transformers import AutoTokenizer, AutoModel
from transformers import TokenClassificationPipeline, AutoModelForTokenClassification

from transformers import logging




# Make dir for Models if not exist
if not os.path.exists("models"):
    os.makedirs("models")

print("Transformer Version: ", transformers.__version__)
print("Torch Version: ", torch.__version__)


# Generate Summaries with BRIO
class BrioSummarizer():
    def __init__(self):
        super().__init__()
        # Instantiate Summarizer Model
        self.IS_CNNDM = True #false uses XSUM dataset & Pegasus Model (Pegasus trained for GAP Sentence Pred as opposed to CNN-DM BRIO)
                        #anecdotally, XSUM returns shorter summaries and CNN-DM returns multi points
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_length = 1024 if self.IS_CNNDM else 512 #CNN/DM dataset has longer summarization len
        self.model_name = 'Yale-LILY/brio-cnndm-uncased'
        self.name = "BRIO-cnndm-uncased"

        logging.set_verbosity_warning() #remove annoying warning, not training here
        logging.set_verbosity_error() #remove really annoying warning

        # Load in Pre-Trained Model (can save models to disk as if it speeds up load-in/inference)
        if self.IS_CNNDM:
            # BART Pre-Trained
            if os.path.isfile("./models/BRIO-Tokenizer.pt"):
                print(f"Loading in Tokenizer for {self.name} from Disk")
                file_to_unpickle = open("./models/BRIO-Tokenizer.pt", "rb")
                self.tokenizer = pickle.load(file_to_unpickle)
            else:
                print(f"Loading in Tokenizer for {self.name} from HF")
                self.tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')

            if os.path.isfile("./models/BRIO-Model.pt"):
                print(f"Loading in Model- {self.name} from Disk")
                file_to_unpickle = open("./models/BRIO-Model.pt", "rb")
                self.model = pickle.load(file_to_unpickle)
                print("") #newline after completing model load
            else:
                print(f"Loading in Model- {self.name} from HF")
                self.model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
                print("") #newline after completing model load
        else:
            # Pegasus Pre-Trained
            if os.path.isfile("./models/BRIO-Tokenizer.pt"):
                print(f"Loading in Tokenizer for {self.name} from Disk")
                file_to_unpickle = open("./models/BRIO-Tokenizer.pt", "rb")
                self.tokenizer = pickle.load(file_to_unpickle)
            else:
                print("Loading in Tokenizer from HF")
                self.tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')

            if os.path.isfile("./models/BRIO-Model.pt"):
                print(f"Loading in Model- {self.name} from Disk")
                file_to_unpickle = open("./models/BRIO-Model.pt", "rb")
                self.model = pickle.load(file_to_unpickle)
                print("") #newline after completing model load
            else:
                print(f"Loading in Model- {self.name} from HF")
                self.model = PegasusForConditionalGeneration.from_pretrained('Yale-LILY/brio-xsum-cased')
                print("") #newline after completing model load

    # Generate Summary
    def summarize(self, article):
        inputs = self.tokenizer([article], max_length=self.max_length, return_tensors="pt", truncation=True)
        inputs.to(self.device) #gofast
        self.model.to(self.device) #gofast
        summary_ids = self.model.generate(inputs["input_ids"])
        output = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output #can format by calling bpf func on this

    # Format Summary Output -- convert sentences into Bullet Points
    def bpf(self, txt):
        bps = []
        for t in txt.split(". "):
            # t = t.strip(".").title() #nice casing for Bullet Points
            t = t.strip(".") #regular sentence casing
            bp = f"- {t}"
            # print(bp)
            bps.append(bp)
        return bps




# Generate Embeddings
class EmbeddingsMiniLM():
    def __init__(self):
        super().__init__()
        # Instantiate Model & Params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2' #name as per hf download
        self.name = 'MiniLM-L6-v2' #natural language name for saving
        self.max_length = 128 #max input len for sentencebert (model based on)

        logging.set_verbosity_warning() #remove warning, not training
        logging.set_verbosity_error() #remove warning, not training

        # Load in Pre-Trained Model & Tokenizer (can save artifacts to disk as if it speeds up load-in/inference)
        if os.path.isfile(f"./models/{self.name}-Tokenizer.pt"):
            print(f"Loading in Tokenizer for {self.name} from Disk")
            file_to_unpickle = open(f"./models/{self.name}-Tokenizer.pt", "rb")
            self.tokenizer = pickle.load(file_to_unpickle)
        else:
            print(f"Loading in Tokenizer for {self.name} from HF")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if os.path.isfile(f"./models/{self.name}-Model.pt"):
            print(f"Loading in Model- {self.name} from Disk")
            file_to_unpickle = open(f"./models/{self.name}-Model.pt", "rb")
            self.model = pickle.load(file_to_unpickle)
            print("") #newline after completing model load
        else:
            print(f"Loading in Model- {self.name} from HF")
            self.model = AutoModel.from_pretrained(self.model_name)
            print("") #newline after completing model load

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Compute Embeddings for Given Text
    def get_embeddings(self, input_text):
        # Tokenize the Input Text
        encoded_input = self.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')

        # Compute Forward Pass (pooled output is embedding)
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Pooling of Output to get Embeddings
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings




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
            print(f"Loading in Tokenizer for {self.name} from Disk")
            file_to_unpickle = open(f"./models/{self.name}-Tokenizer.pt", "rb")
            self.tokenizer = pickle.load(file_to_unpickle)
        else:
            print(f"Loading in Tokenizer for {self.name} from HF")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if os.path.isfile(f"./models/{self.name}-Model.pt"):
            print(f"Loading in Model - {self.name} from Disk")
            file_to_unpickle = open(f"./models/{self.name}-Model.pt", "rb")
            self.model = pickle.load(file_to_unpickle)
            print("") #newline after completing model load
        else:
            print(f"Loading in Model- {self.name} from HF")
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            print("") #newline after completing model load

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




# Download Model Artifacts if Script is run
if __name__ == "__main__":

    # Summarization Model
    BRIO = BrioSummarizer()
    with open("./models/BRIO-Tokenizer.pt", 'wb') as fh:
        pickle.dump(BRIO.tokenizer, fh)

    with open("./models/BRIO-Model.pt", 'wb') as fh:
        pickle.dump(BRIO.model, fh)

    # Embeddings Model
    embedder = EmbeddingsMiniLM()
    with open(f"./models/{embedder.name}-Tokenizer.pt", 'wb') as fh:
        pickle.dump(embedder.tokenizer, fh)

    with open(f"./models/{embedder.name}-Model.pt", 'wb') as fh:
        pickle.dump(embedder.model, fh)

    # Keyphrase Model
    extractor = KeyphraseExtractor(model="ml6team/keyphrase-extraction-kbir-inspec")
    with open(f"./models/{extractor.name}-Tokenizer.pt", 'wb') as fh:
        pickle.dump(extractor.tokenizer, fh)

    with open(f"./models/{extractor.name}-Model.pt", 'wb') as fh:
        pickle.dump(extractor.model, fh)

