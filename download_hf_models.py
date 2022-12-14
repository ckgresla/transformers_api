# CKG Module for HF Model Spin Up

import os
import pickle #for saving binarized models
import torch
import transformers #for checking the version being used
from transformers import BartTokenizer, BartForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration #summarizing module
from transformers import AutoTokenizer, AutoModel
from transformers import LongformerConfig, LongformerModel
from transformers import TokenClassificationPipeline, AutoModelForTokenClassification

from transformers import logging




# Make dir for Models if not exist
if not os.path.exists("models"):
    os.makedirs("models")

print("Transformer Version: ", transformers.__version__)
print("Torch Version: ", torch.__version__)

print("Models Dir:", os.listdir("models"))


# Utility Code
def model_artifacts_download(model):
    # Tokenizer
    with open(f"./models/{model.name}-Tokenizer.pt", 'wb') as fh:
        pickle.dump(model.tokenizer, fh)

   # Model Weights
    with open(f"./models/{model.name}-Model.pt", 'wb') as fh:
        pickle.dump(model.model, fh)
    return




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
            artifact_path = os.path.join("models", f"{self.name}-Tokenizer.pt")
            if os.path.isfile(artifact_path):
                print(f"Loading in Tokenizer for {self.name} from Disk")
                file_to_unpickle = open(artifact_path, "rb")
                self.tokenizer = pickle.load(file_to_unpickle)
            else:
                print(f"Loading in Tokenizer for {self.name} from HF")
                self.tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')

            artifact_path = os.path.join("models", f"{self.name}-Model.pt")
            if os.path.isfile(artifact_path):
                print(f"Loading in Model- {self.name} from Disk")
                file_to_unpickle = open(artifact_path, "rb")
                self.model = pickle.load(file_to_unpickle)
                print("") #newline after completing model load
            else:
                print(f"Loading in Model- {self.name} from HF")
                self.model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
                print("") #newline after completing model load
        else:
            # Pegasus Pre-Trained
            artifact_path = os.path.join("models", f"{self.name}-Tokenizer.pt")
            if os.path.isfile(artifact_path):
                print(f"Loading in Tokenizer for {self.name} from Disk")
                file_to_unpickle = open(artifact_path, "rb")
                self.tokenizer = pickle.load(file_to_unpickle)
            else:
                print("Loading in Tokenizer from HF")
                self.tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')

            artifact_path = os.path.join("models", f"{self.name}-Model.pt")
            if os.path.isfile(artifact_path):
                print(f"Loading in Model- {self.name} from Disk")
                file_to_unpickle = open(artifact_path, "rb")
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


# Generate Embeddings (w MiniLM, Backend for SentenceTransformers)
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
        artifact_path = os.path.join("models", f"{self.name}-Tokenizer.pt")
        if os.path.isfile(artifact_path):
            print(f"Loading in Tokenizer for {self.name} from Disk")
            file_to_unpickle = open(artifact_path, "rb")
            self.tokenizer = pickle.load(file_to_unpickle)
        else:
            print(f"Loading in Tokenizer for {self.name} from HF")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        artifact_path = os.path.join("models", f"{self.name}-Model.pt")
        if os.path.isfile(artifact_path):
            print(f"Loading in Model- {self.name} from Disk")
            file_to_unpickle = open(artifact_path, "rb")
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


# Generate Embeddings (with LongFormers, an interesting Transformer that can handle long docs w/o the typical compute increase)
class EmbeddingsLongformer():
    def __init__(self):
        super().__init__()
        # Instantiate Model & Params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # COULD DOWNLOAD MODEL WEIGHTS LIKE- https://huggingface.co/docs/transformers/v4.22.2/en/model_doc/longformer#transformers.LongformerModel.forward.example
        self.model_name = "allenai/longformer-base-4096" #name as per hf download
        self.name = 'Longformer' #natural language name for saving
        self.max_length = 4096 #max input len for the Base Longformer Model (can fit on an 8GB GPU, larger available but too big)

        logging.set_verbosity_warning() #remove warning, not training
        logging.set_verbosity_error() #remove warning, not training

        # Load in Pre-Trained Model & Tokenizer (can save artifacts to disk as if it speeds up load-in/inference)
        artifact_path = os.path.join("models", f"{self.name}-Tokenizer.pt")
        if os.path.isfile(artifact_path):
            print(f"Loading in Tokenizer for {self.name} from Disk")
            file_to_unpickle = open(artifact_path, "rb")
            self.tokenizer = pickle.load(file_to_unpickle)
        else:
            print(f"Loading in Tokenizer for {self.name} from HF")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        artifact_path = os.path.join("models", f"{self.name}-Model.pt")
        if os.path.isfile(artifact_path):
            print(f"Loading in Model- {self.name} from Disk")
            file_to_unpickle = open(artifact_path, "rb")
            self.model = pickle.load(file_to_unpickle)
            print("") #newline after completing model load
        else:
            print(f"Loading in Model- {self.name} from HF")
            self.model = AutoModel.from_pretrained(self.model_name)
            print("") #newline after completing model load

    # Tokenize Input & Get Embeddings (mean-pooled)
    def get_embeddings(self, input_text):
        encoded = self.tokenizer(input_text, return_tensors="pt", max_length=4096, padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**encoded, output_hidden_states=False) #returns embeddings in output structure (second element, )

        embeddings = output[1] #forward pass provides pooled hidden states
        return embeddings


# Keyphrase Generator (wraps base HF Class) -- different load-in from other models
KBIR_model = "./models/KBIR-Inspec-Model"
KBIR_tokenizer = "./models/KBIR-Inspec-Tokenizer"

class KeyphraseExtractor(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        #super().__init__(model=AutoModelForTokenClassification.from_pretrained(model), tokenizer=AutoTokenizer.from_pretrained(model, model_max_length=512), *args, **kwargs)
        super().__init__(model=AutoModelForTokenClassification.from_pretrained(KBIR_model), tokenizer=AutoTokenizer.from_pretrained(model, model_max_length=512), *args, **kwargs) #special load in for Pipeline
        print("Loading in KBIR, Model from Disk & Tokenizer from HF\n\n") #special load-in sequence

        # Instantiate Model & Params
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "ml6team/keyphrase-extraction-kbir-inspec" #name as per hf download??? https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec
        # Original Paper??? "https://arxiv.org/pdf/2112.08547.pdf"
        self.name = 'KBIR-Inspec' #natural language name for saving (user set)
        self.max_length = 512 #as specified in HF Docs
        self.idx2label = {0: "B", 1: "I", 2: "O"} #mapping for classes in model generation


        logging.set_verbosity_warning() #remove warning, not training
        logging.set_verbosity_error() #remove warning, not training


        # Load in Model Weights (outside of class instantiation, run file as __main__ to get)

    # Util to Get Output from Model into Atomic Keywords (useful format)
    def parse_keywords(self, keyphrases, confidence_threshold=.5):
        relevant_tokens = [] #to hold the relevant spans (after parsing output)
        keyphrases = [(i["word"], i["entity"][0], i["start"], i["end"], i["score"]) for i in keyphrases] #get relevant portions for output

        for word, label, start, end, score in keyphrases:
            word = word.strip("??").strip()
            if label == "B":
                if score >= confidence_threshold:
                    relevant_tokens.append([word])
                prev_end = end #for checking next span
                current_span_score = score
            elif label == "I":
                if len(relevant_tokens) > 0:
                    # Get Consecutive Spans together (changes outputs from [key, phrase, extractor] into [keyphrase, extractor])
                    if prev_end == start:
                        relevant_tokens[len(relevant_tokens) - 1][-1] += word #if current KeyPhrase  span is part of last word
                        prev_end = end #make sure all single word spans are in single word, not spread out as sep tokens
                        current_span_score = (current_span_score + score)/2 #track average confidence score for span
                    else:
                        current_span_score = (current_span_score + score)/2 #track average confidence score for span
                        if current_span_score >= confidence_threshold:
                            relevant_tokens[len(relevant_tokens) - 1].append(word) #appends new word to last list of KeyPhrase Spans
                        prev_end = end #combine all sep token spans into proper words

        # Returned Formatted Output (list of KeyPhrases, not list of lists of relevant tokens)
        relevant_tokens = [" ".join(i) for i in relevant_tokens]
        return relevant_tokens




# Download Model Artifacts if Script is run
if __name__ == "__main__":

    # Summarization Model
    # BRIO = BrioSummarizer()
    # model_artifacts_download(BRIO)

    # Embeddings Model
    embedder = EmbeddingsMiniLM()
    model_artifacts_download(embedder)

    # # Longformer Model
    lf = EmbeddingsLongformer()
    model_artifacts_download(lf)

    # Keyphrase Model
    ##extractor = KeyphraseExtractor(model="ml6team/keyphrase-extraction-kbir-inspec") #old no need use
    # kbirt = AutoTokenizer.from_pretrained("ml6team/keyphrase-extraction-kbir-inspec")
    # kbirt.save_pretrained(KBIR_tokenizer)
    # kbir = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-kbir-inspec")
    # kbir.save_pretrained(KBIR_model)


