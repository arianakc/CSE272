export CUDA_VISIBLE_DEVICES=4
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
# export CUDA_LAUNCH_BLOCKING=1
python3 rerank_cli.py \
  --model vanilla_bert \
  --datafiles data/en.zh.queries.tsv data/zh.documents.tsv \
  --run data/en.zh.test.run \
  --model_weights models/vbert_cli/weights.p \
  --out_path models/vbert_cli/en.zh.test.run