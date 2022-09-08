# Controller for Keyphrase Endpoint

import os
import time
import json

from flask import request, jsonify, make_response
from download_hf_models import KeyphraseExtractor




# Model Wrapper as Instantiated in Load-In
kpe = KeyphraseExtractor(model="ml6team/keyphrase-extraction-kbir-inspec")


# API Resource Wrapper
class PhraseParser():

    def post(self):
        request_data = request.get_json() #expect following vars in the request

        # Key for Text that needs to get summarized
        input_text = request_data.get("input_text")
        #print("INPUT: ", input_text)

        # Check Length, Trim if longer than max context for Model
        print("Lengths of Input & Limit", len(input_text), kpe.max_length)
        if len(input_text) >= kpe.max_length:
            print(f"Input Sequence too long for {kpe.name}, trimming into Sequences")
            input_text = [input_text[i:i+kpe.max_length] for i in range(0, len(input_text), kpe.max_length)] #naive string split approach, cuts input text into 128-character sequences
        else:
            input_text = [input_text] #list of 1 string to summarize

        extracted_keyphrases = set() #set to hold all keyphrases
        sequence_count = 0 #count of sequences to summarize
        total_time = elapsed_time = time.monotonic()
        print(f"\nGenerating Keyphrases for {len(input_text)} Sequences")

        for sequence in input_text:
            start_time = time.monotonic()
            output = kpe(sequence) #forward pass w model
            output = kpe.parse_keywords(output) #get nice tokens from output
            # print(f"OUTPUT @ Sequence {sequence_count}: ", output)
            extracted_keyphrases.update(output) #parse out model output into nice dict of tokens

            end_time = time.monotonic()
            time_diff = end_time - start_time
            elapsed_time += time_diff #to track total time for all summaries
            sequence_count += 1
            print(f"  Sequence {sequence_count} KPE'd in {time_diff:.2f}s")
        print(f"Completed Extraction for Input in: {elapsed_time - total_time:.2f}s")


        # Return Summarized String, if not empty list
        # print(extracted_keyphrases) #prints out the keyphrases extracted to back-end terminal after forward pass (gets returned)
        if len(extracted_keyphrases) != 0:
            extracted_keyphrases = list(extracted_keyphrases) #convert into list, sets not serializable
            extracted_keyphrases = json.dumps(extracted_keyphrases) #json (core util) version
            #extracted_keyphrases = jsonify(extracted_keyphrases) #flask version, weird \n characters in output
            return make_response(extracted_keyphrases, 200)
        else:
            return make_response("Keyphrase Endpoint Error", 400)


    # Get Request Handler, mainly to check server's home dir & health of endpoint
    def get(self):
        print("Get Request Successful - Keyphrase Endpoint")
        output = f"Keyphrase Endpoint- send in JSON data via an 'input_text' key to parse out the Key Phrases\n"
        return make_response(output)


    # Request Handler -- moved in Logic for different request types from Main File
    def request_handler(self):
        if request.method == "GET":
            resp = self.get()
            return resp
        elif request.method == "POST":
            resp = self.post()
            return resp

