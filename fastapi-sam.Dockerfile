FROM python:3.10

## Dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
# RUN apt-get install git-all -y

##  Install FastAPI
COPY fastapi-requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

## Copy src files
COPY src /src

## Run server code
ENTRYPOINT python -u /src/FastAPI-Test-inference_server.py