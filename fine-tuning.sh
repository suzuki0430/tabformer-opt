#!/bin/bash

"===conda activate==="
conda init bash
source /opt/conda/etc/profile.d/conda.sh
conda activate tabformer-opt-sagemaker

"===move directory==="
mv ./tabformer-opt/* .

"===pre-training==="
python tabformer_bert_fine_tuning.py