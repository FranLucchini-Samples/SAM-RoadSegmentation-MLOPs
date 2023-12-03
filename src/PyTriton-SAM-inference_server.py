import torch
from pytriton.decorators import batch
import numpy as np
import torch
from ultralytics import SAM
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
import logging
 


@batch
def infer_fn(**inputs: np.ndarray):
    logger.info(f"CUDA available?: {torch.cuda.is_available()}")

    print(inputs.keys())

    (input1_batch,) = inputs.values()
    logger.info(input1_batch.shape)
    logger.info(type(input1_batch))

    # input_list = [img for img in input1_batch]
    outputs = model(input1_batch[0]) # Calling the Python model inference

    for i, res in enumerate(outputs):
        logger.info(f"Res {i}, len {len(res)}, type {type(res)}")

    print(outputs[0])
    # Return numpy array
    masks = [result.masks.cpu().numpy().data for result in outputs]
    # Returns torch.Tensor
    # masks = outputs[0].masks.cpu().data 
    print(np.array(masks).shape)
    print(np.array([masks]).shape)
    logger.info(f"MASKS LEN {len(masks)} TYPE {type(masks)}")

    batch_masks = [np.array(masks)]

    # Make np.array because intern objects of list must have the .shape attribute
    return batch_masks
    # Does not work: ValueError: Received output tensors with different batch sizes: OUTPUT_1: (41, 400, 640). Expected batch size: 1. 
    # return masks 


# Connecting inference callable with Triton Inference Server


# # Run inference
# model('input/um_000005.png')

logger = logging.getLogger("examples.huggingface_bert_jax.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

logger.info("Hello World! Loading model...")
# Load a model
model = SAM('../weights/sam_b.pt')

# Display model information (optional)
print(model.info())

# print(torch.cuda.get_device_name(0))
# model = torch.nn.Linear(2, 3).to("cuda").eval()

with Triton() as triton:
    logger.info("Binding model.")
    triton.bind(
        model_name="SAM",
        infer_func=infer_fn,
        inputs=[
            Tensor(dtype=np.uint8, shape=(400, 640, 3)),
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128)
    )



    triton.serve()