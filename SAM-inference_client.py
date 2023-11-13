import torch
from pytriton.client import ModelClient
from PIL import Image
import numpy as np

img = Image.open('/home/fran/Learn-MLOps/SAM-RoadSegmentation-MLOPs/input/um_000006.png').resize((640, 400))

np_img = np.expand_dims(
    np.array(img),
    axis=0
)
#.astype(np.float32)
# input1_data = torch.randn(128, 2).cpu().detach().numpy()
print(np_img.shape)


# input1_data = torch.randn(1, 400, 640, 3).cpu().detach().numpy()

with ModelClient("localhost:8000", "SAM") as client:
    result_dict = client.infer_batch(np_img)

print(result_dict)