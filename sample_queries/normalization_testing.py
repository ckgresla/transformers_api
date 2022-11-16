import json
import requests
import numpy as np  #optional, using for testing norms -- not needed


# Configure URL and Desired Endpoint
endpoint = '/summarize' #in base could pick from: ["summarize", "e-mlm", "e-lf", "keyphrase"]
endpoint = '/keyphrase'
endpoint = '/e-lf'
url = "http://localhost:5003" + endpoint


# Sample Text for Request
payload = json.dumps({
 # "input_text": "Container what is container? Package Software into Standardized Units for Development, Shipment and Deployment    A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. A Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and  settings.    Container images become containers at runtime and in the case of Docker containers – images become containers when they run on Docker Engine. Available for both Linux and Windows-based applications, containerized software will always run the same, regardless of the infrastructure. Containers isolate software from its environment and ensure that it works uniformly despite differences for instance between development and staging.    Docker containers that run on Docker Engine:    Standard: Docker created the industry standard for containers, so they could be portable anywhere    Lightweight: Containers share the machine’s OS system kernel and therefore do not require an OS per application, driving higher server efficiencies and reducing server and licensing costs    Secure: Applications are safer in containers and Docker provides the strongest default isolation capabilities in the industry",

  "batch_text": ["TESTING 1", "TESTIN 2", "TESTING 3"],
    "normalize_vecs": True
})
headers = {
  'Content-Type': 'application/json'
}


# POST the Request
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text[0:100])
vec = np.array(response.json())
print(vec.sum(axis=1))
print(vec.shape)

# TEST SAME QUERY w/o Norm
payload = json.dumps({
  #"input_text": "Container what is container? Package Software into Standardized Units for Development, Shipment and Deployment    A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. A Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and  settings.    Container images become containers at runtime and in the case of Docker containers – images become containers when they run on Docker Engine. Available for both Linux and Windows-based applications, containerized software will always run the same, regardless of the infrastructure. Containers isolate software from its environment and ensure that it works uniformly despite differences for instance between development and staging.    Docker containers that run on Docker Engine:    Standard: Docker created the industry standard for containers, so they could be portable anywhere    Lightweight: Containers share the machine’s OS system kernel and therefore do not require an OS per application, driving higher server efficiencies and reducing server and licensing costs    Secure: Applications are safer in containers and Docker provides the strongest default isolation capabilities in the industry"
  "batch_text": ["TESTING 1", "TESTIN 2", "TESTING 3"]
})
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text[0:100])
vec = np.array(response.json())
print(vec.sum(axis=1))
print(vec.shape)
#for i in range(12):
#    response = requests.request("POST", url, headers=headers, data=payload)
#    print(f"response {i} successful")
#    print(response[0:25])

