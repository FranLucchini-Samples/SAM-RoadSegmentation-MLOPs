import requests
from PIL import Image
import numpy as np
import os

local = "localhost"

port = "3000"

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
input_folder = os.path.join(script_path, "..\input")
inputs = []

# TODO: test with th eimage object, not as a numpy array or tensor
for img_fn in [file for file in os.listdir(input_folder) if file.lower().endswith('.png')]:
    print(img_fn)
    img = Image.open(
        os.path.join(
            input_folder,
            img_fn
        )).resize((640, 416))

    np_img = np.expand_dims(
            np.array(img, dtype=np.float16),
            axis=0
        )
    # inputs should be BCHW i.e. shape(1, 3, 640, 400)
    np_img = np_img.transpose((0, 3, 2, 1))

    print(np_img.shape)
    print(np_img[0, 0, :10, :10])
    # Transform np_img to python list
    py_img = np_img.tolist()
    inputs.append(py_img)

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