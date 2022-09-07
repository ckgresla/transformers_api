# Controller for Embedding Endpoint

import os
import time

from flask import request, jsonify, make_response
from download_hf_models import EmbeddingsMiniLM



# From Download Script
MiniLM = EmbeddingsMiniLM()


class Embedify():


    def post(self):
        request_data = request.get_json() #expect following vars in the request

        # Key for Text that needs to get summarized
        input_text = request_data.get("input_text")

        # Check Length, Trim if longer than max context for Model
        print("Lengths of Input & Limit", len(input_text), MiniLM.max_length)
        if len(input_text) >= MiniLM.max_length:
            print(f"Input Sequence too long for {MiniLM.name}, trimming into Sequences")
            input_text = [input_text[i:i+MiniLM.max_length] for i in range(0, len(input_text), MiniLM.max_length)] #naive string split approach, cuts input text into 128-character sequences
        else:
            input_text = [input_text] #list of 1 string to summarize

        embeddings_dict = dict.fromkeys(input_text) #converts list of values into the Keys of a Dictionary with no values
        sequence_count = 0 #count of sequences to summarize
        total_time = elapsed_time = time.monotonic()
        print(f"\nGenerating Embeddings for {len(input_text)} Sequences")

        for sequence in input_text:
            start_time = time.monotonic()
            embeddings_dict[sequence] = MiniLM.get_embeddings(sequence) #re-represent the input text as a embedding/vector, add to Sentence-Embedding Pairs dict

            end_time = time.monotonic()
            time_diff = end_time - start_time
            elapsed_time += time_diff #to track total time for all summaries
            sequence_count += 1
            print(f"  Sequence {sequence_count} embedd-ified in {time_diff:.2f}s")
        print(f"Completed Generation for Input in: {elapsed_time - total_time:.2f}s")


        # Return Summarized String, if not empty list
        if embeddings_dict != {}:
            embeddings_dict = [dict([a, x.tolist()] for a, x in embeddings_dict.items())] #one-liner to convert all tensors into lists for JSON-ifying the Response (may change)
            embeddings_dict = jsonify(embeddings_dict)
            return make_response(embeddings_dict, 200)
        else:
            return make_response("Embedding Endpoint Error", 400)


    # Get Request Handler, mainly to check server's home dir & health of endpoint
    def get(self):
        print("Get Request Successful")
        output = f"This is the embedding endpoint, send in data via a 'input_text' key to get embeddified"
        return output

