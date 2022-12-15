#!/bin/bash

"===current directory==="
pwd

"===list==="
ls -l

"===conda activate==="
conda init bash
source /opt/conda/etc/profile.d/conda.sh
conda activate tabformer-opt-sagemaker

"===move directory==="
mv ./tabformer-opt/* .

"===pre-training==="
python main.py