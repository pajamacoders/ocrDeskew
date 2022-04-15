ARG IMAGE=nvcr.io/nvidia/pytorch
ARG TAG=22.01-py3
FROM ${IMAGE}:${TAG}

RUN pip3 install mlflow einops tqdm