export CUDA_VISIBLE_DEVICES=4
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
# export CUDA_LAUNCH_BLOCKING=1
python3 rerank_mt5_classification.py \
  --model vanilla_mt5 \
  --datafiles data/en.ar.queries.tsv data/ar.documents.tsv \
  --run data/en.ar.test.run \
  --model_weights models/v_mt5_ar/weights.p \
  --out_path models/v_mt5_ar/en.ar.test.run