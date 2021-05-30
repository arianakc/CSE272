export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
# export CUDA_LAUNCH_BLOCKING=1
python3 train_mt5_classification.py \
  --model vanilla_mt5 \
  --datafiles data/en.ar.queries.tsv data/ar.documents.tsv \
  --qrels data/en.ar.qrels \
  --train_pairs data/en.ar.train.pairs \
  --valid_run data/en.ar.dev.run \
  --model_out_dir models/v_mt5_ar