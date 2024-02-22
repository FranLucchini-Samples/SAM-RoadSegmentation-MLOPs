# SAM-RoadSegmentation-MLOPs
Building MLOPs pipelines for automated ML Road Segmentation with SAM

## BentoML Serving

[Source](https://docs.bentoml.org/en/latest/quickstarts/deploy-a-transformer-model-with-bentoml.html?_gl=1*41ck0w*_gcl_au*MTY2NDk4MjEyMy4xNzAxODAxNjE2)

### Download preconfigured models
Starting from BentoML version 1.1.9, you can also use the bentoml.transformers.import_model function to import the model directly without having to load it into memory first, which is particularly useful for large models.
```python
import bentoml

model = "sshleifer/distilbart-cnn-12-6"
task = "summarization"

# Import the model directly without loading it into memory
bentoml.transformers.import_model(
   name=task,
   model_name_or_path=model,
   metadata=dict(model_name=model)
)
```

Then run the script to get the model:
```sh
python download_model.py
```

The model should appear in our BentoML list:
```sh
bentoml models list

# Tag                                    Module                Size       Creation Time
# summarization:5kiyqyq62w6pqnry         bentoml.transformers  1.14 GiB   2023-07-10 11:57:40
```

### Run server

Define a service called `svc` with the `runner` associated to the `summarization`
```python
import bentoml

summarizer_runner = bentoml.models.get("summarization:latest").to_runner()

svc = bentoml.Service(
    name="summarization", runners=[summarizer_runner]
)

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def summarize(text: str) -> str:
    generated = await summarizer_runner.async_run(text, max_length=3000)
    return generated[0]["summary_text"]
```

The file is called `service.py`, which is relevant when we want to run the following command to run the service:
```sh
bentoml serve service:svc
```
The name structure is:
```
bentoml serve <file_name>:<service_variable_name>
```
**NOTE**: you must be inside the folder containing the service file. It does not use paths as service names.

### Build Dockerfile

Put the `BentoML.yaml` file and the requirements file in the same folder as the services needed. Inside that folder run the following command:

```sh
 bentoml build -f BentoML.yaml
```

Run bentoml build in your project directory to build the Bento. All created Bentos are stored in /home/user/bentoml/bentos/ by default.

Next, we run the image with the following command:

```

```

## FastAPI Serving
[Source](https://fastapi.tiangolo.com/#example)

```python
from statistics import mean 
from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
def predict(body: dict):
    res = mean(body['list'])
    return {"res": }

if __name__ == "__main__":
    uvicorn.run("<module-name>:app", host="0.0.0.0", port=8000, reload=True,)
```

This code uses uvicorn to run a server in `localhost:8000` with automatic reload. The code above demonstrates examples of get and post paths.

It is important to note that `uvicorn run` expects the name of the file and the variavle name of the FastAPI server:
```python
# file server_file.py
my_app = FastAPI()
...
    uvicorn.run("server_file:my_app", host="0.0.0.0", port=8000, reload=True,)
```
The flag `reload=True` allows all changes in the code to be applied automatically to the server.


## PyTriton Serving

[Source](https://triton-inference-server.github.io/pytriton/0.4.0/)


```python
import numpy as np
import ModelConfig, Tensor
from pytriton.model_config
from pytriton.triton import Triton
from pytriton.decorators import batch


@batch
def infer_fn(**inputs: np.ndarray):
    '''
    Inference function
    '''
    input1, input2 = inputs.values()
    outputs = model(input1, input2)
    return [outputs]


with Triton() as triton: 
    # Define the connection between triton and the inference function
    triton.bind(
        model_name="MyModel",
        infer_func=infer_fn,
        inputs=[ 
            Tensor(dtype=bytes, shape=(1,)), # sample containing single bytes value
            Tensor(dtype=bytes, shape=(-1,)), # sample containing vector of bytes
            ... 
            ],
        outputs=[ 
            Tensor(dtype=np.float32, shape=(-1,)),
            ],
        config=ModelConfig(max_batch_size=16),
        )

    # Serve the model
    triton.serve()
```

When the `.serve()` method is called on the Triton object, the inference queries can be sent to `localhost:8000/v2/models/MyModel`, and the `infer_fn` is called to handle the inference query.

Run script to start the API serving the model
```sh
python pytriton_inference.py

# I1107 19:16:10.130102 27653 pinned_memory_manager.cc:241] Pinned memory pool is created at '0x7fcd94000000' with size 268435456
# I1107 19:16:10.130481 27653 cuda_memory_manager.cc:107] CUDA memory pool is created on device 0 with size 67108864
# I1107 19:16:10.131326 27653 server.cc:604] 
# +------------------+------+
# | Repository Agent | Path |
# +------------------+------+
# +------------------+------+

# I1107 19:16:10.131377 27653 server.cc:631] 
# +---------+------+--------+
# | Backend | Path | Config |
# +---------+------+--------+
# +---------+------+--------+

# I1107 19:16:10.131435 27653 server.cc:674] 
# +-------+---------+--------+
# | Model | Version | Status |
# +-------+---------+--------+
# +-------+---------+--------+

# I1107 19:16:10.188128 27653 metrics.cc:810] Collecting metrics for GPU 0: NVIDIA GeForce GTX 1060
# I1107 19:16:10.188378 27653 metrics.cc:703] Collecting CPU metrics
# I1107 19:16:10.188609 27653 tritonserver.cc:2415] 
# +----------------------------------+------------------------------------------+
# | Option                           | Value                                    |
# +----------------------------------+------------------------------------------+
# | server_id                        | triton                                   |
# | server_version                   | 2.36.0                                   |
# | server_extensions                | classification sequence model_repository |
# |                                  |  model_repository(unload_dependents) sch |
# |                                  | edule_policy model_configuration system_ |
# |                                  | shared_memory cuda_shared_memory binary_ |
# |                                  | tensor_data parameters statistics trace  |
# |                                  | logging                                  |
# | model_repository_path[0]         | /home/fran/.cache/pytriton/workspace_jey |
# |                                  | c9b6w                                    |
# | model_control_mode               | MODE_EXPLICIT                            |
# | strict_model_config              | 0                                        |
# | rate_limit                       | OFF                                      |
# | pinned_memory_pool_byte_size     | 268435456                                |
# | cuda_memory_pool_byte_size{0}    | 67108864                                 |
# | min_supported_compute_capability | 6.0                                      |
# | strict_readiness                 | 1                                        |
# | exit_timeout                     | 30                                       |
# | cache_enabled                    | 0                                        |
# +----------------------------------+------------------------------------------+

# I1107 19:16:10.190258 27653 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
# I1107 19:16:10.190609 27653 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
# I1107 19:16:10.232215 27653 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
# I1107 19:16:11.263817 27653 model_lifecycle.cc:462] loading: Linear:1
# I1107 19:16:12.532009 27653 python_be.cc:2108] TRITONBACKEND_ModelInstanceInitialize: Linear_0 (CPU device 0)
# I1107 19:16:12.716755 27653 model_lifecycle.cc:817] successfully loaded 'Linear'
```

### Query Served API

#### Server status

```sh
curl -v localhost:8000/v2/health/live

# *   Trying 127.0.0.1:8000...
# * Connected to localhost (127.0.0.1) port 8000 (#0)
# > GET /v2/health/live HTTP/1.1
# > Host: localhost:8000
# > User-Agent: curl/7.81.0
# > Accept: */*
# > 
# * Mark bundle as not supporting multiuse
# < HTTP/1.1 200 OK
# < Content-Length: 0
# < Content-Type: text/plain
# < 
# * Connection #0 to host localhost left intact
```

#### Model status
```sh
curl -v localhost:8000/v2/models/Linear/ready

# *   Trying 127.0.0.1:8000...
# * Connected to localhost (127.0.0.1) port 8000 (#0)
# > GET /v2/models/Linear/ready HTTP/1.1
# > Host: localhost:8000
# > User-Agent: curl/7.81.0
# > Accept: */*
# > 
# * Mark bundle as not supporting multiuse
# < HTTP/1.1 200 OK
# < Content-Length: 0
# < Content-Type: text/plain
# < 
# * Connection #0 to host localhost left intact

```


#### Inference

```sh
curl -X POST \
  -H "Content-Type: application/json"  \
  -d @input/input.json \
  localhost:8000/v2/models/Linear/infer

# {
#     "id":"0",
#     "model_name":"Linear",
#     "model_version":"1",
#     "outputs":
#     [
#         {
#             "name":"OUTPUT_1",
#             "datatype":"FP32","shape":[1,3],
#             "data":[0.7278651595115662,0.5738978981971741,0.2889821231365204]
#         }
#     ]
# }
```

#### Inference with image file
```sh
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/home/fran/Learn-MLOps/SAM-RoadSegmentation-MLOPs/input/um_000007.png" \
  localhost:8000/v2/models/SAM/infer

```

## Docker

### PyTriton

```sh
docker build --force-rm -f pytriton-sam.Dockerfile -t sam-pytriton .
```

```sh
docker run --rm -it \
--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v <absolute-path-to-local-folder>:/src \
sam-pytriton
```

```sh
docker run --rm -it \
--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v /home/fran/Learn-MLOps/SAM-RoadSegmentation-MLOPs/src:/src \
sam-pytriton
```

Map port 8080 on the Docker host to TCP port 80 in the container.

```sh
docker run --rm -it -p 8000:8000 --gpus all \
sam-pytriton
```

### FastAPI
```sh
docker build --force-rm -f fastapi-sam.Dockerfile -t sam-fastapi .
```

```sh
docker run --rm -it \
--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v <absolute-path-to-local-folder>:/src \
sam-fastapi
```

```sh
docker run --rm -it \
--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v /home/fran/Learn-MLOps/SAM-RoadSegmentation-MLOPs/src:/src \
sam-fastapi
```

Map port 8080 on the Docker host to TCP port 80 in the container.

```sh
docker run --rm -it -p 8000:8000 --gpus all \
sam-fastapi
```

## Sphinx's Docs

## Start

```bash
pip install -r docs/requirements.txt
sphinx-quickstart docs # NOTE: Run only if docs folder is not created yet!
```

## Autogenerate Docs

Use the Autodoc Extension to create new docs automatically.

### Edit Path
In [`docs/conf.py`](docs\conf.py) add the folder containing the file to document.

```python
sys.path.insert(0, os.path.abspath('..'))
# Add new path starting from the root using ../
sys.path.insert(0, os.path.abspath('../<folder_relative_path>'))

```

### Edit docstring

Following the documentation of [`sphinx.ext.autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html), include the new modules or classes to document in the [`docs/index.rst`]():

```rst
Welcome to sam-roads's documentation!
=====================================

.. automodule:: <new_file_module_path>
   :members:
```

### Build html files
Linux/MacOS
```bash
cd docs | make html
```

Windows
```bash
docs\make.bat html
```
