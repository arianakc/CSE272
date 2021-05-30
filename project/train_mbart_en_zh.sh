export CUDA_VISIBLE_DEVICES=4
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
# export CUDA_LAUNCH_BLOCKING=1
python3 train_mbart.py \
  --model vanilla_mbart \
  --datafiles data/en.zh.queries.tsv data/zh.documents.tsv \
  --qrels data/en.zh.qrels \
  --train_pairs data/en.zh.train.pairs \
  --valid_run data/en.zh.dev.run \
  --model_out_dir models/v_mbart_zh