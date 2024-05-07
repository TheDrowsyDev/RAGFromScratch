#!/bin/bash

model=/home/dev/data/models/bge-reranker-base

docker run -p 8002:80 \
-v $model:/model \
ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 \
--model-id /model