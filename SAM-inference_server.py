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

    (input1_batch,) = inputs.values()

    '''
    AttributeError: 'Results' object has no attribute 'shape'. See valid attributes below.

    A class for storing and manipulating inference results.

    Args:
        orig_img (numpy.ndarray): The original image as a numpy array.
        path (str): The path to the image file.
        names (dict): A dictionary of class names.
        boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
        masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
        probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
        keypoints (List[List[float]], optional): A list of detected keypoints for each object.

    Attributes:
        orig_img (numpy.ndarray): The original image as a numpy array.
        orig_shape (tuple): The original image shape in (height, width) format.
        boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
        masks (Masks, optional): A Masks object containing the detection masks.
        probs (Probs, optional): A Probs object containing probabilities of each class for classification task.
        keypoints (Keypoints, optional): A Keypoints object containing detected keypoints for each object.
        speed (dict): A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
        names (dict): A dictionary of class names.
        path (str): The path to the image file.
        _keys (tuple): A tuple of attribute names for non-empty attributes.
    
    '''

    # logger.info(input1_batch.shape)

    outputs = model(input1_batch[0]) # Calling the Python model inference
    # np.asarray(outputs)
    # outputs_batch = outputs.numpy()
    # logger.info(outputs.numpy().shape)
    # logger.info(f"TYPE {type(outputs[0])}, LEN {len(outputs[0])}")
    for i, res in enumerate(outputs):
        logger.info(f"Res {i}, len {len(res)}, type {type(res)}")

    print(outputs[0])
    # Return numpy array
    masks = [result.masks.cpu().numpy().data for result in outputs]
    # Returns torch.Tensor
    # masks = outputs[0].masks.cpu().data 
    print(np.array(masks).shape)
    print([np.array(masks)].shape)
    logger.info(f"MASKS LEN {len(masks)} TYPE {type(masks)}")

    return [np.array(masks)]


# Connecting inference callable with Triton Inference Server


# # Run inference
# model('input/um_000005.png')

logger = logging.getLogger("examples.huggingface_bert_jax.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

logger.info("Hello World! Loading model...")
# Load a model
model = SAM('weights/sam_b.pt')

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

# if __name__ == "__main__":
#     main()
