export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
# export CUDA_LAUNCH_BLOCKING=1
python3 rerank_mbart.py \
  --model vanilla_mbart \
  --datafiles data/en.zh.queries.tsv data/zh.documents.tsv \
  --run data/en.zh.test.run \
  --model_weights models/mbart_zh/weights.p \
  --out_path models/mbart_zh/en.zh.test.run