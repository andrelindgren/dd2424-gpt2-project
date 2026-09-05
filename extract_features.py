#!/usr/bin/env python3
'''
Stores GPT-2 last-token features for faster last-linear-layer training.
Used with train_head.py

uv run python extract_features.py --csv data/ids-sst-train.csv --flag train --filepath sst_train_features.pt --batch_size 8 --use_gpu --run_name sst-train

uv run python extract_features.py --csv data/ids-sst-dev.csv --flag valid --filepath sst_dev_features.pt --batch_size 8 --use_gpu --run_name sst-dev

'''
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from models.gpt2 import GPT2Model
from classifier import load_data, SentimentDataset, seed_everything


def extract(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    if args.flag == 'train':
        data, _ = load_data(args.csv, args.flag)
    elif args.flag == 'valid':
        data = load_data(args.csv, args.flag)

    dataset = SentimentDataset(data, argparse.Namespace(max_length=args.max_length))

    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                        collate_fn=dataset.collate_fn)

    model = GPT2Model.from_pretrained().to(device)
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    if args.use_gpu:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    # Want to save all tokens, labels and ids for the entire epoch
    feats, labels, sent_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)

            # Forward pass
            last_token = model(b_ids, b_mask)['last_token']

            feats.append(last_token.cpu())
            labels.extend(batch['labels'].tolist())
            sent_ids.extend(batch['sent_ids'])

    extract_seconds = time.perf_counter() - start
    peak_mem_bytes = torch.cuda.max_memory_allocated() if args.use_gpu else 0

    save_info = {
        'features': torch.cat(feats),
        'labels': torch.tensor(labels),
    }
    torch.save(save_info, args.filepath)
    print(f'Total Features: {len(sent_ids)}')
    print(f'saved to {args.filepath}')

    if args.run_name:
        os.makedirs('results', exist_ok=True)
        with open(f'results/{args.run_name}_extract.csv', 'w') as f:
            f.write('num_examples,extract_seconds,peak_mem_bytes\n')
            f.write(f'{len(sent_ids)},{extract_seconds},{peak_mem_bytes}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--flag')
    parser.add_argument('--filepath')
    parser.add_argument('--seed', type=int, default=11711)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--run_name', type=str, default=None)
    args = parser.parse_args()
    seed_everything(args.seed)
    extract(args)