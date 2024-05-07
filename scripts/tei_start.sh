#!/bin/bash

model=/home/dev/data/models/nomic-embed-text-v1.5
volume=$PWD/tei-data

docker run -p 8000:80 \
-v $model:/model \
ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 \
--model-id /model