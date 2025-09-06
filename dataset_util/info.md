# shard ds simple usage
```
python shard_dataset.py \
  --dataset roneneldan/TinyStories \
  --splits train,validation \
  --tokenizer facebook/bart-base \
  --out_dir tiny_llama_bin \
  --prefix tiny \
```

jwk ds prep test
```
python dataset_util/shard_dataset.py \
  --dataset roneneldan/TinyStories \
  --splits train,validation \
  --tokenizer gpt2 \
  --out_dir /p/vast1/kirchenb/.cache/ldlm/binary_datasets/tinystories_gpt2 \
  --prefix tiny

python dataset_util/shard_dataset.py \
  --dataset mintujupally/ROCStories \
  --splits train,test \
  --tokenizer gpt2 \
  --out_dir /p/vast1/kirchenb/.cache/ldlm/binary_datasets/rocstories_gpt2 \
  --prefix roc
```