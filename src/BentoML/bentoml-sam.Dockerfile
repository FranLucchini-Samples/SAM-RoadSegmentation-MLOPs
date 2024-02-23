FROM python:3.10

# # Dpendencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
# RUN apt-get install git-all -y

##  Install BentoML
COPY bentoml-requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

## Copy src files
COPY src /src
COPY weights /weights

## Run server code
# ENTRYPOINT watch -n 1 nvidia-smi
# ENTRYPOINT <run server>
