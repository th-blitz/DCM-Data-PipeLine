FROM python:3.9-slim

LABEL MAINTAINER="preethamrakshith11@gmail.com"

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN mkdir -p /myapp
WORKDIR /myapp

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install pydicom
RUN pip install numpy
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install pandas
RUN pip install glob2
RUN pip install ipykernel
RUN pip install pylibjpeg
RUN pip install pylibjpeg-libjpeg==1.3.1
RUN pip install scikit-image
RUN pip install matplotlib
RUN pip install jupyter notebook



