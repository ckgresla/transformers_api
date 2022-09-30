# Transformers API
A simple docker container to spin up HF Models as API Endpoints -- primarily for use in other tasks.

**This container is currently running with torch 1.10.2 and the cudatoolkit 11.3 (cu113), set to install via the `Dockerfile`* 

(i.e, config the install of Torch based on your compute environment)



<BR>
<BR>

## Startup
Makefiles are brilliant, the included one let's you build the container ez with a simple `make build && make start` at the command line (in the container dir).
- build : docker command to build the container
- start : docker command to serve the API with the base dir as a shared volume
- test : health-check query to confirm the container is up (if on same machine)

The docker container will create the environment from scratch, but if you'd rather use the API with your own env -- one could just run `python main.py` in this folder if the correct env is sourced (runs flask on your local machine instead of inside the container)


<BR>
<BR>


## Sample Queries
Running everything w curl can get a bit pesky, setting up a testing environment in Postman is easy enough (POST to url with header; Content-Type="application/json" and body; {"input_text": "your text to be processed"}) but I have included sample queries for cURL and Python in the "sample_queries" dir -- just need specify the url of the container (in case it is same machine, localnet, or somewhere in the ether)


Serving this to your friends is easy (albeit a bit sketchy) with [ngrok](https://ngrok.com/), the following command should do the trick (if you have a free account) if you feel the need to let your Homies & Homettes make use of your compute:

`ngrok http 5003`

^connects port 5003 to the world via ngrok's servers


<BR>
<BR>


## Currently Available Endpoints
All being served on "localhost:5003":
- "/summarize" (BRIO-based Summarization, check out ['Yale-LILY/brio-cnndm-uncased'](https://huggingface.co/Yale-LILY/brio-cnndm-uncased) on HF for Details)
    - returns a bullet-point formatted string
- "/e-mlm" or "/e-lf" (Multiple Endpoints for Embedding Generation, see ["sentence-transformers/paraphrase-MiniLM-L6-v2"](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) or ["allenai/longformer-base-4096"](https://huggingface.co/docs/transformers/v4.22.2/en/model_doc/longformer#longformer) on HF for Details)
    - returns an Array of Arrays (i.e, returned_array[0][N][dim]) wherein;
      - First array is always [0] (the list that holds the subsequent values)
      - N=number of embeddings generated (number of batches if using "batch_text" or 1 if using "input_text")
      - dim=the dimension of the embedding vector (as per the model spec)
- "/keyphrase" (KBIR-based Keyphrase Extraction, check out ["ml6team/keyphrase-extraction-kbir-inspec"](https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec) on HF for information-s)
    - returns an array of keyphrases
    - **the Keyphrase model uses the HF TokenClassificationPipeline which results in having to download the model from HF each time**


<BR>
<BR>


## Adding New Models
It isn't too bad to add additional models as endpoints, all we need do is the following: 
1. Add a class for Downloading the Model and Running the Desired Functions to `download_hf_models.py`
2. Create a controller for the Endpoint in the `controllers` folder (handle the GET | POST requests)
3. Add a URL for the Endpoint (referencing the created controller) in the `main.py` file
4. Update the documentation ;) (readme & sample_queries)
5. Enjoy


<BR>
<BR>

