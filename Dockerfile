FROM nvcr.io/nvidia/tensorflow:20.02-tf2-py3

EXPOSE 8888/tcp
WORKDIR /home
COPY . .
RUN python -m pip install .