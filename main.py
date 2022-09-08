# Flask API
from settings import *
from flask import Flask, jsonify, request




# Instantiate App
app = Flask(__name__)


# Define Routes per Model Endpoint/Controller
# Model responses are JSON-ified in the respective controller files

# Health Handler
@app.route("/health", methods=['GET','POST'])
def health():
    if request.method=='GET':
        return dict(greeting="This is the Transformers Endpoint, Welcome!"), 200
    else:
        return jsonify({'Error':"Sorry, the '/health' endpoint accepts GETs only Comrade"})

# Summarization Handler
if TEXT_SUMMARIZATION:
    from controllers.summarizers import SummaryGenerator
    SG = SummaryGenerator()
    app.add_url_rule("/summarize", "summarize", SG.request_handler, methods=["GET", "POST"])

# Embedding Handler
from controllers.embeddings import Embedify
if EMBEDDINGS_GENERATION:
    EG = Embedify()
    app.add_url_rule("/embedding", "embedding", EG.request_handler, methods=["GET", "POST"])

# Keyphrase Handler
if KEYPHRASE_EXTRACTION:
    from controllers.keyphrases import PhraseParser
    KP = PhraseParser()
    app.add_url_rule("/keyphrase", "keyphrase", KP.request_handler, methods=["GET", "POST"])


# Run App on Configured Port
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT_TO_SERVE) #port specified in settings


