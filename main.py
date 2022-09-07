# Flask API
# import settings
from flask import Flask, jsonify, request

from controllers.summarizers import SummaryGenerator
from controllers.keyphrases import PhraseParser
from controllers.embeddings import Embedify


# Instantiate App
app = Flask(__name__)
PORT_TO_SERVE = 5003



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
SG = SummaryGenerator()
app.add_url_rule("/", "summarize", SG)
@app.route("/summarize", methods=['GET','POST'])
def m1():
    if request.method=='GET':
        resp = SG.get()
        return resp
    else:
        resp = SG.post()
        return resp

# Keyphrase Handler
KP = PhraseParser()
@app.route("/keyphrase", methods=['GET','POST'])
def m2():
    if request.method=='GET':
        resp = KP.get()
        return resp
    else:
        resp = KP.post()
        return resp

# Embedding Handler
E = Embedify()
@app.route("/embedding", methods=['GET','POST'])
def m3():
    if request.method=='GET':
        resp = E.get()
        return resp
    else:
        resp = E.post()
        return resp




# Run App on Configured Port
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT_TO_SERVE)


