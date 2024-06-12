import json
import os
import csv
from pathlib import Path
from PIL import Image

import open_clip
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
#import wandb

from src.CLIP_fine_tuning.CLIP_utils import load_model, random_seed, dataset, evaluate_BiVLC

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',                 default=None)
    parser.add_argument("--seed", type=int,         default=None) # 42
    parser.add_argument('--clip-model',             default="ViT-B-32")
    parser.add_argument('--pretrained',             default="openai")
    parser.add_argument('--dataset',                default='evaluation')
    parser.add_argument('--model_cache_dir', type=Path,   default='')
    parser.add_argument('--model-checkpoint', type=Path,       default=None)
    parser.add_argument('--batch-size', type=int,   default=1000)
    parser.add_argument('--save-path', type=Path,   default='./results')
    parser.add_argument("--project_name", type=str, default='CLIP_BiVLC')
    parser.add_argument("--run_name", type=str,     default='ViT-B-32')


    args = parser.parse_args()

    #os.system('wandb login')

    config = {"model":args.clip_model, "pretrained":args.pretrained, "fine-tuned":args.run_name, "seed":torch.random.initial_seed()}  

    #wandb.init(project = args.project_name, name = args.run_name, config=config)

    if args.seed:
        random_seed(seed = args.seed)
    print(torch.random.initial_seed())
    
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, preprocess = load_model(args)

    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loaded checkpoint from {args.model_checkpoint}')

    data, collate_fn = dataset(args, preprocess)
    test_dataloader = DataLoader(data , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=False)
    print(data)
    print(f'Evaluating {args.clip_model} fine-tuned in {args.run_name} for {len(data)} instances...')   
    # Evaluation
    model.eval() 
    text_score = []
    image_score = []
    group_score = []
    
    text_score_i0 = []
    text_score_i1 = []
    image_score_c0 = []
    image_score_c1 = []

    with torch.inference_mode():
        pbar = tqdm(test_dataloader)
        for batch in pbar:
            text_s, image_s, group_s, text_s_i0, text_s_i1, image_s_c0, image_s_c1 = evaluate_BiVLC(args, batch, model, tokenizer)
            text_score.extend(text_s)
            image_score.extend(image_s)
            group_score.extend(group_s)
            text_score_i0.extend(text_s_i0)
            text_score_i1.extend(text_s_i1)
            image_score_c0.extend(image_s_c0)
            image_score_c1.extend(image_s_c1) 
                               
    #wandb.log({'Text score': 100*(sum(text_score)/len(data)), 'Image score': 100*(sum(image_score)/len(data)), 'Group score': 100*(sum(group_score)/len(data)),'Text score_i0': 100*(sum(text_score_i0)/len(data)), 'Text score_i1': 100*(sum(text_score_i1)/len(data)), 'Image score_c0': 100*(sum(image_score_c0)/len(data)), 'Image score_c1': 100*(sum(image_score_c1)/len(data))})        
    print(f'{args.run_name} checkpoint OVERALL scores:   Text score: {100*(sum(text_score)/len(data))}, Image score: {100*(sum(image_score)/len(data))}, Group score: {100*(sum(group_score)/len(data))}, Text score_i0: {100*(sum(text_score_i0)/len(data))}, Text score_i1: {100*(sum(text_score_i1)/len(data))}, Image score_c0: {100*(sum(image_score_c0)/len(data))}, Image score_c1: {100*(sum(image_score_c1)/len(data))}')

    df = pd.DataFrame(data)
    df['text_score_i0'] = [tensor.item() for tensor in text_score_i0]
    df['text_score_i1'] = [tensor.item() for tensor in text_score_i1]
    df['image_score_c0'] = [tensor.item() for tensor in image_score_c0]
    df['image_score_c1'] = [tensor.item() for tensor in image_score_c1]
    df['text_score'] = [tensor.item() for tensor in text_score]
    df['image_score'] = [tensor.item() for tensor in image_score]
    df['group_score'] = [tensor.item() for tensor in group_score]
    
    df.to_csv(os.path.join(args.save_path, f'BiVLC_{args.run_name}.csv'), index = False, quotechar='"', quoting=csv.QUOTE_ALL)

    df_replace = df[df['type'] == 'replace']
    df_swap = df[df['type'] == 'swap']
    df_add = df[df['type'] == 'add']

    print(f"{args.run_name} checkpoint REPLACE ({len(df_replace)}) scores:   Text score: {100*(sum(df_replace['text_score'])/len(df_replace))}, Image score: {100*(sum(df_replace['image_score'])/len(df_replace))}, Group score: {100*(sum(df_replace['group_score'])/len(df_replace))}, Text score_i0: {100*(sum(df_replace['text_score_i0'])/len(df_replace))}, Text score_i1: {100*(sum(df_replace['text_score_i1'])/len(df_replace))}, Image score_c0: {100*(sum(df_replace['image_score_c0'])/len(df_replace))}, Image score_c1: {100*(sum(df_replace['image_score_c1'])/len(df_replace))}")
    print(f"{args.run_name} checkpoint SWAP ({len(df_swap)}) scores:   Text score: {100*(sum(df_swap['text_score'])/len(df_swap))}, Image score: {100*(sum(df_swap['image_score'])/len(df_swap))}, Group score: {100*(sum(df_swap['group_score'])/len(df_swap))}, Text score_i0: {100*(sum(df_swap['text_score_i0'])/len(df_swap))}, Text score_i1: {100*(sum(df_swap['text_score_i1'])/len(df_swap))}, Image score_c0: {100*(sum(df_swap['image_score_c0'])/len(df_swap))}, Image score_c1: {100*(sum(df_swap['image_score_c1'])/len(df_swap))}")
    print(f"{args.run_name} checkpoint ADD ({len(df_add)}) scores:   Text score: {100*(sum(df_add['text_score'])/len(df_add))}, Image score: {100*(sum(df_add['image_score'])/len(df_add))}, Group score: {100*(sum(df_add['group_score'])/len(df_add))}, Text score_i0: {100*(sum(df_add['text_score_i0'])/len(df_add))}, Text score_i1: {100*(sum(df_add['text_score_i1'])/len(df_add))}, Image score_c0: {100*(sum(df_add['image_score_c0'])/len(df_add))}, Image score_c1: {100*(sum(df_add['image_score_c1'])/len(df_add))}")
