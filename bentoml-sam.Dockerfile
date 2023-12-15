FROM python:3.10

# # Dpendencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
# RUN apt-get install git-all -y

##  Install pytriton
# RUN pip install -U nvidia-pytriton==0.4.0 ultralytics==8.0 opencv-python==4.8.1.78 Pillow==10.0.1
COPY bentoml-requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

## Copy src files
COPY src /src
COPY weights /weights

## Run server code
# ENTRYPOINT watch -n 1 nvidia-smi
# ENTRYPOINT <run server>
