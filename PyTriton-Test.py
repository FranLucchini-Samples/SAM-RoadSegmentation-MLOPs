import torch
from pytriton.decorators import batch
import numpy as np
import torch


print(torch.cuda.get_device_name(0))
model = torch.nn.Linear(2, 3).to("cuda").eval()


@batch
def infer_fn(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to("cuda")
    output1_batch_tensor = model(input1_batch_tensor) # Calling the Python model inference
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]


from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# Connecting inference callable with Triton Inference Server
with Triton() as triton:
    triton.bind(
        model_name="Linear",
        infer_func=infer_fn,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128)
    )

    triton.serve()