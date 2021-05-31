export CUDA_VISIBLE_DEVICES=4
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
# export CUDA_LAUNCH_BLOCKING=1
python3 rerank_mt5_classification.py \
  --model vanilla_mt5 \
  --datafiles data/en.zh.queries.tsv data/zh.documents.tsv \
  --run data/en.zh.test.run \
  --model_weights models/mt5_c_zh/weights.p \
  --out_path models/mt5_c_zh/en.zh.test.run