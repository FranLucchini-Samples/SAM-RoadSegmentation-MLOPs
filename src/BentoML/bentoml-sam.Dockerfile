# FROM python:3.10
FROM nvcr.io/nvidia/pytorch:24.01-py3

# # Dpendencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

##  Install BentoML
COPY src/BentoML/bentoml-requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

## Copy src files
COPY src/BentoML/BentoML_Test_inference_server.py /BentoML_Test_inference_server.py
COPY src/BentoML/weights /weights

## Run server code
# ENTRYPOINT watch -n 1 nvidia-smi
WORKDIR /
ENTRYPOINT bentoml serve -p 5000 BentoML_Test_inference_server:svc
