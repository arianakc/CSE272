export CUDA_VISIBLE_DEVICES=4
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
# export CUDA_LAUNCH_BLOCKING=1
python3 rerank_mbert.py \
  --model vanilla_bert \
  --datafiles data/en.ar.queries.tsv data/ar.documents.tsv \
  --run data/en.ar.test.run \
  --model_weights models/mbert_ar/weights.p \
  --out_path models/mbert_ar/en.ar.test.run