import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON


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
        import torch

        self.model = torch.hub.load("ultralytics/yolov5:v6.2", "yolov5s")

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

        # Config inference settings
        self.inference_size = 320

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs):
        """_summary_

        Args:
            input_imgs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Return predictions only
        results = self.model(input_imgs, size=self.inference_size)
        return results.pandas().xyxy

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
async def inference(input_img):
    """_summary_

    Args:
        input_img (_type_): _description_

    Returns:
        _type_: _description_
    """
    batch_ret = await yolo_v5_runner.inference.async_run([input_img])
    return { "predictions": [1,2,3] }
