# SAM-RoadSegmentation-MLOPs
Building MLOPs pipelines for automated ML Road Segmentation with SAM

## PyTriton Serving

[Source](https://triton-inference-server.github.io/pytriton/0.4.0/)



```python {"skip": true}
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

## Query Served API

### Server status

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

### Model status
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


### Inference

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
