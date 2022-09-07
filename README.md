# Transformers API

a simple docker container to spin up HF Models as API Endpoints, Save any Model Weights/Artifacts to a Dir in the Container w Support for GPU Inference and make requests to the endpoint


This container is currently running with torch 1.10.2 and the cudatoolkit 11.3 (cu113), set to install via the `docker/setup.sh` file


## Available Endpoints
all on "localhost:5003", had to solve a recv issue when trying to run on a different port:
- "/summarize" (BRIO-based Summarization, returns a bullet-point formatted string)
- "/embedify" (base endpoint for generating embeddings, need config to return task specific items)
- "/keyphrase" (keyphrase extraction)


