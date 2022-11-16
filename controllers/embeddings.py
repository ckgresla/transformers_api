# Controller for Embedding Endpoint

import os
import time

import torch
from flask import request, jsonify, make_response
from download_hf_models import EmbeddingsMiniLM
from download_hf_models import EmbeddingsLongformer


# From Download Script -- if extending class no need to instantiate the Models Here
# MiniLM = EmbeddingsMiniLM()
# LF = EmbeddingsLongformer()


# Wrappers for the Embedding Models (run them as API Endpoints)
class MiniLM_Endpoint(EmbeddingsMiniLM):

    def post(self):
        request_data = request.get_json() #expect following vars in the request

        # Parse JSON input for Single Sequence or Multi-Sequences
        if request_data.get("batch_text"):
            input_text = request_data.get("batch_text") #multi texts to embed (array of strings)
            batch = True
            input_text = [input_text] #will iter through
            print(f"Parsing a Batch of Text: {len(input_text)}")
        else:
            input_text = request_data.get("input_text") #single text to embed (string, singular)
            batch = False
            input_text = [input_text]

        # Compute Embeddings
        print(f"\nGenerating Embeddings for {len(input_text)} Sequences")
        total_time = elapsed_time = time.monotonic()
        embeddings_array = []
        sequence_count = 0

        if batch:
            # Expecting List of Lists here (each contains the string to embed)
            for sequence in input_text:
                start_time = time.monotonic()
                embeddings_array.append(self.get_embeddings(sequence)) #compute embeddings and add to return item

                end_time = time.monotonic()
                time_diff = end_time - start_time
                elapsed_time += time_diff #to track total time for all summaries
                sequence_count += 1
                # print(f"  Sequence {sequence_count} embedd-ified in {time_diff:.2f}s")
            print(f"Completed Embedding Compute for Input in: {elapsed_time - total_time:.2f}s")
        else:
            embeddings_array.append(self.get_embeddings(input_text))
            print(f"Completed Embedding Compute for Input in: {time.monotonic() - total_time:.2f}s")

        # If no embeddings, throw error -- else return array of embeddings
        print(f"Number of Embeddings in Return Array: {len(embeddings_array)}")
        if len(embeddings_array) != 0:
            embeddings_array = jsonify([i.tolist() for i in embeddings_array])
            return make_response(embeddings_array, 200)
        else:
            return make_response("Embedding Endpoint Error", 400)


    # Get Request Handler, mainly to check server's home dir & health of endpoint
    def get(self):
        print("Get Request Successful")
        output = f"Embedding Endpoint- send in JSON data via a 'input_text' key to get embeddified\n"
        return output

    # Request Handler -- moved in Logic for different request types from Main File
    def request_handler(self):
        if request.method == "GET":
            resp = self.get()
            return resp
        elif request.method == "POST":
            resp = self.post()
            return resp


class Longformer_Endpoint(EmbeddingsLongformer):

    def post(self):
        request_data = request.get_json() #expect following vars in the request

        # Parse JSON input for Single Sequence or Multi-Sequences
        if request_data.get("batch_text"):
            input_text = request_data.get("batch_text") #multi texts to embed (array of strings)
            batch = True
            input_text = [input_text] #will iter through
            print(f"Parsing a Batch of Text: {len(input_text)}")
        else:
            input_text = request_data.get("input_text") #single text to embed (string, singular)
            batch = False
            input_text = [input_text]

        # Normalize Embeddings if in Request Params (bool)
        if request_data.get("normalize_vecs"):
            normalize_vecs = True
        else:
            normalize_vecs = False

        # Compute Embeddings
        print(f"\nGenerating Embeddings for {len(input_text)} Sequences")
        total_time = elapsed_time = time.monotonic()
        embeddings_array = []
        sequence_count = 0

        if batch:
            # Expecting List of Lists here (each contains the string to embed)
            for sequence in input_text:
                start_time = time.monotonic()
                embeddings_array.append(self.get_embeddings(sequence)) #compute embeddings and add to return item

                end_time = time.monotonic()
                time_diff = end_time - start_time
                elapsed_time += time_diff #to track total time for all summaries
                sequence_count += 1
                # print(f"  Sequence {sequence_count} embedd-ified in {time_diff:.2f}s")
            print(f"Completed Embedding Compute for Input in: {elapsed_time - total_time:.2f}s")
        else:
            embeddings_array.append(self.get_embeddings(input_text))
            print(f"Completed Embedding Compute for Input in: {time.monotonic() - total_time:.2f}s")

        # If no embeddings, throw error -- else return array of embeddings
        print(f"Number of Embeddings in Return Array: {len(embeddings_array)}")
        if len(embeddings_array) != 0:
            if normalize_vecs:
                print("NORMALIZING VECS")
                embeddings_array = torch.cat(embeddings_array) #concatenate the list of embeddings (basically tranforms the list into a tensor)
                embeddings_array = torch.nn.functional.normalize(embeddings_array, dim=1)
            embeddings_array = jsonify([i.tolist() for i in embeddings_array])
            return make_response(embeddings_array, 200)
        else:
            return make_response("Embedding Endpoint Error", 400)


    # Get Request Handler, mainly to check server's home dir & health of endpoint
    def get(self):
        print("Get Request Successful")
        output = f"Embedding Endpoint- send in JSON data via a 'input_text' key to get embeddified\n"
        return output

    # Request Handler -- moved in Logic for different request types from Main File
    def request_handler(self):
        if request.method == "GET":
            resp = self.get()
            return resp
        elif request.method == "POST":
            resp = self.post()
            return resp

