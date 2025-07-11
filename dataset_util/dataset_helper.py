from multiprocessing.spawn import prepare
from multiprocess import cpu_count
import itertools
import os
import json
from typing import Optional

from datasets import load_dataset, Value 
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator
from datasets import DatasetDict, Dataset
import numpy as np
import torch

from datasets import load_from_disk

from .collator import DataCollatorForBartDenoisingLM, DataCollatorForLatentVAE, DataCollatorForLatentVAET5

NUM_PROC = min(cpu_count(),64) # probably seeing hyperthreading, don't clobber the node

class PrecomputedLatentDataset(Dataset):
    """
    A dataset that loads pre-computed latent tensors from a single memory-mapped .npy file.
    """
    def __init__(self, latent_dir: str):
        self.latent_dir = latent_dir
        
        # Load metadata
        metadata_path = os.path.join(self.latent_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {self.latent_dir}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.num_samples = self.metadata['num_samples']
        
        # Memory-map the latents file
        latents_path = os.path.join(self.latent_dir, "latents.npy")
        if not os.path.exists(latents_path):
            raise FileNotFoundError(f"latents.npy not found in {self.latent_dir}")
        
        # Get shape and dtype from metadata
        dtype = np.dtype(self.metadata['dtype'])
        shape = (self.metadata['num_samples'], self.metadata['seq_len'], self.metadata['latent_dim'])
            
        self.latents = np.memmap(latents_path, dtype=dtype, mode='r', shape=shape)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # The latents are already pre-computed and stored.
        # We just need to fetch the correct one.
        latent = self.latents[idx]
        return {"input_latents": torch.from_numpy(latent)}


def exists(x):
    return x is not None


def get_dataset(dataset_name, metadata=False, synthetic_train_path=None, shard_size=None):
    if dataset_name == 'roc':
        roc_data_path = 'datasets/ROCstory'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(roc_data_path, f'roc_{split}.json') for split in ['train', 'valid']}, num_proc=NUM_PROC)
        dataset = process_roc_dataset(dataset)
    elif dataset_name == 'ag_news':
        dataset = load_dataset('pietrolesci/ag_news', 'original', num_proc=NUM_PROC)
        train_ds = dataset['train']
        train_val_ds = train_ds.train_test_split(test_size=1000, seed=42)
        train_val_ds['valid'] = train_val_ds['test']
        train_val_ds['test'] = dataset['test']
        dataset = process_ag_news_dataset(train_val_ds)
    elif dataset_name == 'xsum':
        dataset = load_dataset('xsum', num_proc=NUM_PROC)
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_xsum_dataset(dataset)
    elif dataset_name == 'qqp':
        qqp_data_path = 'datasets/qqp'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(qqp_data_path, f'{split}.jsonl') for split in ['train', 'valid', 'test']}, num_proc=NUM_PROC)
        dataset = process_qqp_dataset(dataset)
    elif dataset_name == 'wmt14-de-en':
        dataset = load_dataset('wmt14', 'de-en', num_proc=NUM_PROC)
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'de-en')
    elif dataset_name == 'wmt14-de-de':
        dataset = load_dataset('wmt14', 'de-en', num_proc=NUM_PROC)
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'de-de')
    elif dataset_name == 'wmt14-en-de':
        dataset = load_dataset('wmt14', 'de-en', num_proc=NUM_PROC)
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'en-de')
    elif dataset_name == 'wmt14-en-en':
        dataset = load_dataset('wmt14', 'de-en', num_proc=NUM_PROC)
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'en-en')
    elif dataset_name == "fineweb-edu_10b":
        dataset = process_fineweb_edu(shard_size=shard_size)
    # elif dataset_name == "fineweb_100b":
    #     dataset = process_fineweb_100b(shard_size=shard_size)
    elif dataset_name == "fineweb_350b":
        dataset = process_fineweb_350b(shard_size=shard_size)
    elif dataset_name == "tiny_stories":
        dataset = process_tiny_stories()
    else:
        raise NotImplementedError
    return dataset


def process_fineweb_edu(
    split_ratio = 0.1,
    # output_dir="/scratch/gpfs/ashwinee/datasets/fineweb_edu_splits/", # Directory to save/load splits
    output_dir=f"/p/vast1/{os.getenv('USER','FIXME')}/.cache/ldlm/datasets/fineweb_edu_splits/", # Directory to save/load splits
    force_resplit=False, # Option to force re-splitting even if files exist
    shard_size: Optional[int] = None # Number of samples to use for a small shard
):
    os.makedirs(output_dir, exist_ok=True)
    split_dataset_path = os.path.join(output_dir, f"fineweb_edu_split_valid{int(split_ratio*100)}")

    if os.path.exists(split_dataset_path) and not force_resplit:
        print(f"Loading pre-split dataset from {split_dataset_path}")
        loaded_ds_dict = DatasetDict.load_from_disk(split_dataset_path)
    else:
        dataset = load_dataset("EleutherAI/fineweb-edu-dedup-10b", num_proc=NUM_PROC)
        print(f"Splitting dataset: fineweb-edu_10b. This may take some time for large datasets.")
        dataset = dataset['train']
        split_datasets = dataset.train_test_split(
            test_size = split_ratio,
            seed = 42
        )

        final_ds_dict = DatasetDict({
            'train': split_datasets['train'],
            'valid': split_datasets['test']
        })
        
        print(f"Dataset split complete. Saving to {split_dataset_path}...")
        final_ds_dict.save_to_disk(split_dataset_path, num_proc=NUM_PROC)
        print("Split dataset saved to disk.")
        loaded_ds_dict = final_ds_dict

    if shard_size is not None:
        print(f"Creating a small shard of size {shard_size} for testing.")
        loaded_ds_dict['train'] = loaded_ds_dict['train'].select(range(shard_size))
        # Also shard the validation set proportionally
        val_shard_size = max(1, int(shard_size * split_ratio))
        loaded_ds_dict['valid'] = loaded_ds_dict['valid'].select(range(val_shard_size))

    return loaded_ds_dict


def process_fineweb_350b(
    split_ratio = 0.1,
    # output_dir="/scratch/gpfs/ashwinee/datasets/fineweb_edu_splits/", # Directory to save/load splits
    output_dir=f"/p/vast1/{os.getenv('USER','FIXME')}/.cache/ldlm/datasets/fineweb_350b_splits/", # Directory to save/load splits
    force_resplit=False, # Option to force re-splitting even if files exist
    shard_size: Optional[int] = None # Number of samples to use for a small shard
):
    # If this function does fail (eg. hang), its possible it just needs to be run standalone.
    # I suspect, but do not know, that something wonky is happening between slurm/flux, accelerate,
    # and whatever mp pool datasets tries to spin up for the download at this moment.
    # This fn can be imported from the root and run arg-less in an interactive session.
    # TODO: make a "preproc" top level script that takes the ds name and just does this.

    os.makedirs(output_dir, exist_ok=True)
    split_dataset_path = os.path.join(output_dir, f"fineweb_350b_split_valid{int(split_ratio*100)}")

    if os.path.exists(split_dataset_path) and not force_resplit:
        print(f"Loading pre-split dataset from {split_dataset_path}")
        loaded_ds_dict = DatasetDict.load_from_disk(split_dataset_path)
    else:
        print(f"Loading fineweb_350b. If downloading, this could take some time or fail.")
        dataset = load_dataset("HuggingFaceFW/fineweb", "sample-350BT", num_proc=NUM_PROC)
        print(f"Splitting dataset: fineweb_350b. This may take some time for large datasets.")
        dataset = dataset['train']
        split_datasets = dataset.train_test_split(
            test_size = split_ratio,
            seed = 42
        )

        final_ds_dict = DatasetDict({
            'train': split_datasets['train'],
            'valid': split_datasets['test']
        })
        
        print(f"Dataset split complete. Saving to {split_dataset_path}...")
        final_ds_dict.save_to_disk(split_dataset_path, num_proc=NUM_PROC)
        print("Split dataset saved to disk.")
        loaded_ds_dict = final_ds_dict

    if shard_size is not None:
        print(f"Creating a small shard of size {shard_size} for testing.")
        loaded_ds_dict['train'] = loaded_ds_dict['train'].select(range(shard_size))
        # Also shard the validation set proportionally
        val_shard_size = max(1, int(shard_size * split_ratio))
        loaded_ds_dict['valid'] = loaded_ds_dict['valid'].select(range(val_shard_size))

    return loaded_ds_dict

def process_tiny_stories(
    split_ratio = 0.1,
    output_dir="./tiny_stories_splits/", # Directory to save/load splits
    force_resplit=False # Option to force re-splitting even if files exist
):
    os.makedirs(output_dir, exist_ok=True)
    split_dataset_path = os.path.join(output_dir, f"tiny_stories_split_valid{int(split_ratio*100)}")

    if os.path.exists(split_dataset_path) and not force_resplit:
        print(f"Loading pre-split dataset from {split_dataset_path}")
        loaded_ds_dict = DatasetDict.load_from_disk(split_dataset_path)
        return loaded_ds_dict
    else:
        dataset = load_dataset("roneneldan/TinyStories", num_proc=NUM_PROC)
        print(f"Splitting dataset: TinyStories. This may take some time for large datasets.")
        dataset = dataset['train']
        split_datasets = dataset.train_test_split(
            test_size = split_ratio,
            seed = 42
        )

        final_ds_dict = DatasetDict({
            'train': split_datasets['train'],
            'valid': split_datasets['test']
        })
        
        print(f"Dataset split complete. Saving to {split_dataset_path}...")
        final_ds_dict.save_to_disk(split_dataset_path, num_proc=NUM_PROC)
        print("Split dataset saved to disk.")
        return final_ds_dict


def process_roc_dataset(dataset):
    def extract_roc_text(example):
        text = example['text']
        assert text[:2] == '["'
        assert text[-2:] == '"]'
        sentences = text[2:-2]
        return {'text': sentences}
    dataset = dataset.map(extract_roc_text, )
    dataset = dataset.shuffle(seed=42)
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

def process_ag_news_dataset(dataset):
    def process_ag_news_text(example):
        # return {'text': PreTrainedTokenizerBase.clean_up_tokenization(f'Title: {example["title"]}<pad> Description: {example["description"]}'.strip()), 'label':example['label']-1}
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["description"].strip()), 'label':example['label']-1}
    dataset = dataset.map(process_ag_news_text, remove_columns=['title', 'description', 'class'])
    return dataset

def process_xsum_dataset(dataset):
    def process_xsum_text(example):
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["summary"].strip()), 'context':PreTrainedTokenizerBase.clean_up_tokenization(example["document"].strip())}
    dataset = dataset.map(process_xsum_text, remove_columns=['summary', 'document', 'id'])
    dataset = dataset.shuffle(seed=42)
    return dataset

def process_qqp_dataset(dataset):
    def process_qqp_text(example):
        dict_example = json.loads(example['text'])
        dict_example['text'] = dict_example['trg']
        dict_example['context'] = dict_example['src']
        del dict_example['trg']
        del dict_example['src']
        return dict_example
    dataset = dataset.map(process_qqp_text, )
    dataset = dataset.shuffle(seed=42)
    return dataset

def process_wmt14_dataset(dataset, lang_pair):
    def process_wmt14_text(example, lang_pair):
        source, target = lang_pair.split('-')
        assert source in ['de', 'en']
        assert target in ['de', 'en']

        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example['translation'][target].strip()), 'context':PreTrainedTokenizerBase.clean_up_tokenization(example['translation'][source].strip())}
    dataset = dataset.map(process_wmt14_text, fn_kwargs={'lang_pair': lang_pair}, remove_columns=['translation'])
    dataset = dataset.shuffle(seed=42)
    return dataset

def parse_metadata(metadata):
    if type(metadata) == list:
        return ' | '.join(metadata)
    elif type(metadata) == float:
        return 'Positive' if metadata > 0.5 else 'Negative'


def get_dataloader(args, dataset, model_config, tokenizer, max_seq_len, mode='diffusion', shuffle=True, context_tokenizer=None):
    def tokenization(example):
        if mode == 'diffusion' and args.dataset_name in {'xsum', 'qqp',  'wmt14-en-de', 'wmt14-de-en'}:
            assert context_tokenizer is not None
            source = example['context']
            target = example['text']

            if args.dataset_name in {'qqp', 'wmt14-en-de', 'wmt14-de-en'}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len)
            elif args.dataset_name in {'xsum',}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len*4)
            else:
                raise NotImplementedError

            model_inputs = tokenizer(text_target=target, padding="max_length", truncation=True, max_length=max_seq_len)
            
            # Add model target to model inputs
            for k in cond_inputs.keys():
                model_inputs[f'cond_{k}'] = cond_inputs[k]

            return model_inputs
        else:
            text = example["text"]
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)
    
    collate_fn=DataCollatorForBartDenoisingLM(tokenizer, model_config.decoder_start_token_id)
    # if 'mbart' in args.enc_dec_model:
    #     collate_fn=default_data_collator
    # elif 'bart' in args.enc_dec_model:
        
    # else:
    #     raise NotImplementedError
    
    if args.dataset_name in {'xsum', 'qqp'} or 'wmt14' in args.dataset_name:
        dataset = dataset.map(tokenization, remove_columns=['text', 'context'], batched=True, num_proc=None)
    else:
        dataset = dataset.map(tokenization, remove_columns='text')
            
    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.train_bs,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 4
        )
    return dl


def get_dataloader_lvae(cfg,
                        dataset, 
                        tokenizer, 
                        max_seq_len, 
                        mode='diffusion', 
                        shuffle=True, 
                        context_tokenizer=None,
                        processed_data_path="./tokenized_ds"):
    def tokenization(example):
        text = example["text"]
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)
    
    collate_fn = DataCollatorForLatentVAE(tokenizer)

    # Create a unique path for the tokenized dataset to avoid vocab mismatch.
    sanitized_tokenizer_name = tokenizer.name_or_path.replace("/", "__")
    # processed_data_path = f"tokenized_ds/{cfg.data.dataset_name}/{sanitized_tokenizer_name}_seqlen{max_seq_len}"
    processed_data_path = f"/p/vast1/{os.getenv('USER','FIXME')}/.cache/ldlm/tokenized_ds/{cfg.data.dataset_name}/{sanitized_tokenizer_name}_seqlen{max_seq_len}"

    if os.path.exists(processed_data_path):
        print(f"Loading tokenized dataset from disk: {processed_data_path}")
        tokenized_dataset = load_from_disk(processed_data_path)
    else:
        print("Tokenizing dataset...")        
        # since more compute bound I think, no hardcap like NUM_PROC above for downloads
        # but always leave a smidge
        num_cores = int(cpu_count() * 0.9)
        tokenized_dataset = dataset.map(
            tokenization,
            batched=True,
            num_proc=num_cores,
            remove_columns=['text', "id", "metadata"] 
            
        )
        print(f"Saving tokenized dataset to disk: {processed_data_path}")
        tokenized_dataset.save_to_disk(processed_data_path, num_proc=NUM_PROC)
    
    batch_size = cfg.training.train_bs if shuffle else cfg.training.eval_bs
    dl = DataLoader(
            tokenized_dataset,
            collate_fn = collate_fn,
            batch_size = batch_size,
            shuffle = shuffle,
            pin_memory = True,
            num_workers = os.cpu_count()
        )
    return dl


def get_dataloader_lvae_t5(
                        args,
                        dataset,
                        model_config, 
                        tokenizer, 
                        max_seq_len, 
                        mode='diffusion', 
                        shuffle=True, 
                        context_tokenizer=None,
                        use_precomputed_latents=False,
                        precomputed_latent_path=None,
                        batch_size=128):

    if use_precomputed_latents:
        if precomputed_latent_path is None:
            raise ValueError("precomputed_latent_path must be provided when use_precomputed_latents is True")
        
        dataset = PrecomputedLatentDataset(latent_dir=precomputed_latent_path)

        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            collate_fn=default_data_collator, # Use default collator as we are returning dicts
            batch_size=batch_size,
            # num_workers=args.num_workers,
            num_workers=os.cpu_count() // 2,
            pin_memory=True,
        )
        return dataloader

    # --- This part below is for on-the-fly tokenization if not using precomputed latents ---

    # Adjust the cache path to be unique for the tokenizer model
    tokenizer_name = tokenizer.name_or_path.replace("/", "__")

    def tokenization(example):
        text = example["text"]
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)
    
    collate_fn = DataCollatorForLatentVAET5(tokenizer, model_config.decoder_start_token_id)
    
    # Create a unique path for the tokenized dataset to avoid vocab mismatch.
    processed_data_path = f"tokenized_ds/{args.dataset_name}/{tokenizer_name}_seqlen{max_seq_len}"

    if os.path.exists(processed_data_path):
        print(f"Loading tokenized dataset from disk: {processed_data_path}")
        tokenized_dataset = load_from_disk(processed_data_path)
    else:
        print("Tokenizing dataset...")        
        num_cores = max(os.cpu_count() // 4, 2) 
        tokenized_dataset = dataset.map(
            tokenization,
            batched=True,
            num_proc=num_cores,
            remove_columns=['text', 'id', 'metadata'] 
            
        )
        print(f"Saving tokenized dataset to disk: {processed_data_path}")
        tokenized_dataset.save_to_disk(processed_data_path, num_proc=NUM_PROC)
    
    dl = DataLoader(
            tokenized_dataset,
            collate_fn = collate_fn,
            batch_size = args.train_bs,
            shuffle = shuffle,
            pin_memory = True,
            num_workers = max(os.cpu_count() // 2, 4)
        )
    return dl


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader for .bin files

import glob
from pathlib import Path

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int, sequence_length: int, tokenizer=None, start_file_idx: int = 0):
    """
    Generator that yields batches from .bin files in the format expected by LatentVAE.
    
    Args:
        filename_pattern: glob pattern for .bin files
        batch_size: total batch size across all ranks
        rank: current process rank
        world_size: total number of processes
        sequence_length: sequence length for each sample
        tokenizer: tokenizer instance for validation (optional)
        start_file_idx: index of file to start from (for resuming training)
    
    Yields:
        dict with 'input_ids', 'attention_mask', and 'labels' keys
    """
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert len(files) > 0, f"No files found matching pattern: {filename_pattern}"
    assert batch_size % world_size == 0, f"Batch size {batch_size} must be divisible by world size {world_size}"
    
    local_batch_size = batch_size // world_size
    
    # Start from the specified file index
    if start_file_idx >= len(files):
        start_file_idx = 0
    
    # Create iterator starting from the specified file
    files_from_start = files[start_file_idx:] + files[:start_file_idx]
    file_iter = itertools.cycle(files_from_start)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    
    while True:
        if pos + batch_size * sequence_length + 1 >= len(tokens):
            try:
                tokens, pos = _load_data_shard(next(file_iter)), 0
            except StopIteration:
                # Reset file iterator for multi-epoch training
                file_iter = iter(files)
                tokens, pos = _load_data_shard(next(file_iter)), 0
        
        # Extract local batch for this rank
        start_idx = pos + rank * local_batch_size * sequence_length
        batch_tokens = tokens[start_idx:start_idx + local_batch_size * sequence_length]
        
        # Reshape to (local_batch_size, sequence_length)
        batch_tokens = batch_tokens.view(local_batch_size, sequence_length)

        # Convert to CUDA tensors
        inputs = batch_tokens.to(device="cuda", dtype=torch.int32, non_blocking=True)
        
        # Create attention mask (all ones since we're not using padding in this format)
        attention_mask = torch.ones_like(inputs, dtype=torch.int64)
        
        # Create batch dict in the format expected by the model
        batch = {
            "input_ids": inputs,
            "attention_mask": attention_mask,
        }
        
        pos += batch_size * sequence_length
        yield batch

def _num_tokens_in_shard(file: Path):
    """
    Similar to _load_data_shard, but does not load the tokens into memory.
    """
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    return num_tokens

def wind_data_generator(cfg, train_bin_pattern, step, rank, world_size, tokenizer=None):
    """
    Wind the data generator to the correct position for resuming training.
    
    Args:
        cfg: Configuration object
        train_bin_pattern: Pattern for training bin files
        step: Current training step to resume from
        rank: Current process rank
        world_size: Total number of processes
        tokenizer: Tokenizer for validation (optional)
    
    Returns:
        Data generator starting from the correct position
    """
    import glob
    
    # Calculate total tokens consumed up to this step
    train_bs = cfg.training.train_bs
    grad_accumulate = cfg.training.grad_accumulate
    max_seq_len = cfg.model.max_seq_len
    
    # Total tokens consumed = step * batch_size * grad_accumulate * world_size * seq_len
    total_tokens_consumed = step * train_bs * grad_accumulate * world_size * max_seq_len
    
    # Get all training files
    files = [Path(file) for file in sorted(glob.glob(train_bin_pattern))]
    
    if not files:
        raise ValueError(f"No files found matching pattern: {train_bin_pattern}")
    
    print(f"Found {len(files)} training files")
    
    # Wind forward by counting tokens in each file without loading them
    tokens_consumed = 0
    file_idx = 0
    
    while tokens_consumed < total_tokens_consumed and file_idx < len(files):
        try:
            shard_size = _num_tokens_in_shard(files[file_idx])
            if tokens_consumed + shard_size <= total_tokens_consumed:
                # Consume the entire shard
                tokens_consumed += shard_size
                file_idx += 1
            else:
                # This shard contains our target position, but we'll start from the next one
                file_idx += 1
                break
        except Exception as e:
            print(f"Error reading shard {files[file_idx]}: {e}")
            file_idx += 1
    
    # If we've gone through all files, wrap around to the beginning
    if file_idx >= len(files):
        file_idx = 0
    
    print(f"Wound data generator to step {step} (consumed {tokens_consumed} tokens)")
    print(f"Starting from file {files[file_idx] if file_idx < len(files) else 'beginning (wrapped around)'}")
    
    # Create a new data generator starting from the calculated position
    return distributed_data_generator(
        filename_pattern=train_bin_pattern,
        batch_size=train_bs,
        rank=rank,
        world_size=world_size,
        sequence_length=max_seq_len,
        tokenizer=tokenizer,
        start_file_idx=file_idx
    )

def wind(ckpt_path, step=None):
    config = OmegaConf.load(os.path.join(ckpt_path, "config.json"))
    train_seq_len = config.train_seq_len
    effective_steps = config.max_iters if step is None else step
    total_tokens_to_consume = effective_steps * train_seq_len
    files = [Path(file) for file in sorted(glob.glob(config.train_files))]
    
    # Wind forward by counting tokens in each file without loading them
    tokens_consumed = 0
    file_idx = 0
    
    while tokens_consumed < total_tokens_to_consume and file_idx < len(files):
        shard_size = _num_tokens_in_shard(files[file_idx])
        if tokens_consumed + shard_size <= total_tokens_to_consume:
            # Consume the entire shard
            tokens_consumed += shard_size
            file_idx += 1
        else:
            # This shard contains our target position, but we'll start from the next one
            file_idx += 1
            break
    
    # If we've gone through all files, wrap around to the beginning
    if file_idx >= len(files):
        file_idx = 0
        
    print(f"Wound data generator to step {effective_steps} (consumed {tokens_consumed} tokens)")
    print(f"Starting from file {files[file_idx]}")
    
    # Create a new data generator starting from the next shard
    return data_generator(config.train_files, train_seq_len, start_file_idx=file_idx)


def get_dataloader_lvae_bin(cfg, filename_pattern: str, rank: int, world_size: int, tokenizer=None, start_file_idx: int = 0):
    """
    Creates a data generator for LVAE training using .bin files.
    
    Args:
        cfg: Configuration object
        filename_pattern: glob pattern for .bin files (e.g., "data/train_*.bin")
        rank: current process rank
        world_size: total number of processes
        tokenizer: tokenizer instance for validation (optional)
        start_file_idx: index of file to start from (for resuming training)
    
    Returns:
        Generator that yields batches
    """
    return distributed_data_generator(
        filename_pattern=filename_pattern,
        batch_size=cfg.training.train_bs,
        rank=rank,
        world_size=world_size,
        sequence_length=cfg.model.max_seq_len,
        tokenizer=tokenizer,
        start_file_idx=start_file_idx
    )

def get_val_dataloader_lvae_bin(cfg, filename_pattern: str, rank: int, world_size: int, tokenizer=None, start_file_idx: int = 0):
    """
    Creates a validation data generator for LVAE training using .bin files.
    
    Args:
        cfg: Configuration object
        filename_pattern: glob pattern for validation .bin files (e.g., "data/val_*.bin")
        rank: current process rank
        world_size: total number of processes
        tokenizer: tokenizer instance for validation (optional)
        start_file_idx: index of file to start from (for resuming training)
    
    Returns:
        Generator that yields batches
    """
    return distributed_data_generator(
        filename_pattern=filename_pattern,
        batch_size=cfg.training.eval_bs,
        rank=rank,
        world_size=world_size,
        sequence_length=cfg.model.max_seq_len,
        tokenizer=tokenizer,
        start_file_idx=start_file_idx
    )

def wind_data_generator_cfm(cfg, train_bin_pattern, step, rank, world_size, tokenizer=None):
    """
    Wind the data generator for CFM training to the correct position for resuming training.
    
    Args:
        cfg: Configuration object for CFM training
        train_bin_pattern: Pattern for training bin files
        step: Current training step to resume from
        rank: Current process rank
        world_size: Total number of processes
        tokenizer: Tokenizer for validation (optional)
    
    Returns:
        Data generator starting from the correct position
    """
    import glob
    
    # Calculate total tokens consumed up to this step
    # CFM trainer uses different config structure
    train_bs = cfg.training.train_bs
    grad_accumulate = cfg.training.gradient_accumulate_every
    max_seq_len = getattr(cfg.model, 'max_seq_len', 1024)  # CFM gets this from model config
    
    # Total tokens consumed = step * batch_size * grad_accumulate * world_size * seq_len
    total_tokens_consumed = step * train_bs * grad_accumulate * world_size * max_seq_len
    
    # Get all training files
    files = [Path(file) for file in sorted(glob.glob(train_bin_pattern))]
    
    if not files:
        raise ValueError(f"No files found matching pattern: {train_bin_pattern}")
    
    print(f"Found {len(files)} training files")
    
    # Wind forward by counting tokens in each file without loading them
    tokens_consumed = 0
    file_idx = 0
    
    while tokens_consumed < total_tokens_consumed and file_idx < len(files):
        try:
            shard_size = _num_tokens_in_shard(files[file_idx])
            if tokens_consumed + shard_size <= total_tokens_consumed:
                # Consume the entire shard
                tokens_consumed += shard_size
                file_idx += 1
            else:
                # This shard contains our target position, but we'll start from the next one
                file_idx += 1
                break
        except Exception as e:
            print(f"Error reading shard {files[file_idx]}: {e}")
            file_idx += 1
    
    # If we've gone through all files, wrap around to the beginning
    if file_idx >= len(files):
        file_idx = 0
    
    print(f"Wound CFM data generator to step {step} (consumed {tokens_consumed} tokens)")
    print(f"Starting from file {files[file_idx] if file_idx < len(files) else 'beginning (wrapped around)'}")
    
    # Create a new data generator starting from the calculated position
    return distributed_data_generator(
        filename_pattern=train_bin_pattern,
        batch_size=train_bs,
        rank=rank,
        world_size=world_size,
        sequence_length=max_seq_len,
        tokenizer=tokenizer,
        start_file_idx=file_idx
    )
