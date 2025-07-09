from multiprocessing.spawn import prepare
from multiprocess import cpu_count
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
