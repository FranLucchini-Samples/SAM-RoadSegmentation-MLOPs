import requests
from PIL import Image
import numpy as np
import os
import cv2
import json
import base64

local = "localhost"

port = "5000"

url = f"http://{local}:{port}"


# Basic GET request
response = requests.get(f'{url}/healthz')
print('GET healthz:', response)

# response = requests.get(f'{url}/livez')
# print('GET livez:', response)

# response = requests.get(f'{url}/readyz')
# print('GET readyz:', response)

# response = requests.get(f'{url}/metrics')
# print('GET metrics:', response)
script_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_path, "..\..\input")
inputs = []

# TODO: test with th eimage object, not as a numpy array or tensor
for img_fn in [file for file in os.listdir(input_folder) if file.lower().endswith('.png')]:
    print(img_fn)
    # Read image with cv2
    img = cv2.imread(
        os.path.join(
            input_folder,
            img_fn
        )
    )

    # Convert to jpg using cv2
    _, buffer = cv2.imencode('.jpg', img)
    # Decode buffer to base64
    enc_img = base64.b64encode(buffer).decode('utf-8')
    # Append to inputs
    inputs.append(enc_img)

body = {
        "instances": inputs,
        "parameters": {"param1": 1.54}
    }

# POST request with body
response = requests.post(
    f"{url}/inference",
    json=body,
)
print(response.json())

# Dump response to json file with indent=2
with open('response.json', 'w') as f:
    json.dump(response.json(), f, indent=2)