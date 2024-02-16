import torch
from ultralytics import SAM
import bentoml
# from bentoml.io import NumpyNdarray
from bentoml.io import JSON
import numpy as np
# from bentoml.io import Image


class Yolov5Runnable(bentoml.Runnable):
    """BentoML Runnable for a model

    Args:
        bentoml (_type_): _description_

    Returns:
        _type_: _description_
    """
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        """Initialization
        """

        # Load a model
        self.model = SAM('sam_b.pt')

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

        # Config inference settings
        # self.inference_size = 320

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs):
        """_summary_

        Args:
            input_imgs (list<np.array>): _description_

        Returns:
            _type_: _description_
        """
        # SAM model does not support batched inferences yet
        results = []
        for im in input_imgs:
            # SAM ouput: tuple of pred_masks, pred_scores, pred_bboxes
            pred_masks, pred_scores, pred_bboxes = self.model(im)
            results.append({
                "pred_masks": pred_masks,
                "pred_scores": pred_scores,
                "pred_bboxes": pred_bboxes
            })
        # print(results)
        # results = [res.tojson() for res in self.model(np.asarray(input_imgs[0]))]
        # return results
        return "a"

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_imgs):
        """_summary_

        Args:
            input_imgs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Return images with boxes and labels
        return self.model(input_imgs, size=self.inference_size).render()


yolo_v5_runner = bentoml.Runner(Yolov5Runnable, max_batch_size=30)
svc = bentoml.Service("yolo_v5_demo", runners=[yolo_v5_runner])

# NOTE: Healthcheck funtion handled by Bento

@svc.api(input=JSON(), output=JSON())
async def inference(body):
    """_summary_

    Args:
        body (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print(body)
    # Transform each image in body istances to a torch tensor
    images = [torch.tensor(img) for img in body['instances']]
    # print(images.shape)
    batch_ret = await yolo_v5_runner.inference.async_run(images)
    return { "results": batch_ret }
