import bentoml
from bentoml.io import Image
from bentoml.io import PandasDataFrame


class Yolov5Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
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
        # Return predictions only
        results = self.model(input_imgs, size=self.inference_size)
        return results.pandas().xyxy

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_imgs):
        # Return images with boxes and labels
        return self.model(input_imgs, size=self.inference_size).render()


yolo_v5_runner = bentoml.Runner(Yolov5Runnable, max_batch_size=30)
svc = bentoml.Service("yolo_v5_demo", runners=[yolo_v5_runner])

# NOTE: Healthcheck funtion handled by Bento

@svc.api(input=JSON(), output=JSON())
async def inference(input_img):
    batch_ret = await yolo_v5_runner.inference.async_run([input_img])
    return { "predictions": [1,2,3] }
