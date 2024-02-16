"""Dummy client for BentoML
"""
from bentoml.client import Client
import requests
import os
from PIL import Image
import numpy as np

client = Client.from_url("http://localhost:3000")

# # Basic GET request
# response = requests.get(f'{url}/healthcheck')
# print(response.json())

body = {
        "instances": [],
        "parameters": {"param1": 1.54},
    }

inputs = []
input_folder = 'input/'
for img_fn in [file for file in os.listdir(input_folder) if file.lower().endswith('.png')]:
    print(img_fn)
    img = Image.open(
        os.path.join(
            input_folder,
            img_fn
        )).resize((640, 400))
    
    np_img = np.expand_dims(
        np.array(img),
        axis=0
    )

    print(np_img.shape)
    inputs.append(np_img)

body["instances"] = inputs

res = client.call("inference", body)
print(res)
