#!/bin/bash

model=/home/dev/data/models/mistral-7b-instruct-v0.2

docker run -p 8001:80 \
--gpus all --shm-size 1g \
-v $model:/model \
ghcr.io/huggingface/text-generation-inference:1.4 \
--model-id /model \
--quantize eetq