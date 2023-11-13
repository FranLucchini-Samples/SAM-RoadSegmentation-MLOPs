import torch
from pytriton.client import ModelClient, AsyncioModelClient
from PIL import Image
import numpy as np
import os


input_folder = '/home/fran/Learn-MLOps/SAM-RoadSegmentation-MLOPs/input/'
inputs = []
for img_fn in [file for file in os.listdir(input_folder) if file.lower().endswith('.png')]:
    print(img_fn)
    img = Image.open(
        os.path.join(
            input_folder,
            img_fn
        )).resize((640, 400))

    np_img = np.array(img)
    # np.expand_dims(
    #     np.array(img),
    #     axis=0
    # )

    print(np_img.shape)
    inputs.append(np_img)


# input1_data = torch.randn(1, 400, 640, 3).cpu().detach().numpy()

with ModelClient("localhost:8000", "SAM") as client:
    result_dict = client.infer_batch(np.array(inputs))

print(result_dict)
for key in result_dict.keys():
    print(key)
    res = result_dict[key]
    print(type(res), res.shape)