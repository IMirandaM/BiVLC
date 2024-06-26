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
    I2T = []
    T2I = []
    group_score = []
    
    Ipos_2T = []
    Ineg_2T = []
    Tpos_2I = []
    Tneg_2I = []

    with torch.inference_mode():
        pbar = tqdm(test_dataloader)
        for batch in pbar:
            I2T_s, T2I_s, group_s, Ipos_2T_s, Ineg_2T_s, Tpos_2I_s, Tneg_2I_s = evaluate_BiVLC(args, batch, model, tokenizer)
            I2T.extend(I2T_s)
            T2I.extend(T2I_s)
            group_score.extend(group_s)
            Ipos_2T.extend(Ipos_2T_s)
            Ineg_2T.extend(Ineg_2T_s)
            Tpos_2I.extend(Tpos_2I_s)
            Tneg_2I.extend(Tneg_2I_s) 
                               
    #wandb.log({'I2T': 100*(sum(I2T)/len(data)), 'T2I': 100*(sum(T2I)/len(data)), 'Group score': 100*(sum(group_score)/len(data)),'Ipos_2T': 100*(sum(Ipos_2T)/len(data)), 'Ineg_2T': 100*(sum(Ineg_2T)/len(data)), 'Tpos_2I': 100*(sum(Tpos_2I)/len(data)), 'Tneg_2I': 100*(sum(Tneg_2I)/len(data))})        
    print(f'{args.run_name} checkpoint OVERALL scores:   I2T: {100*(sum(I2T)/len(data))}, T2I: {100*(sum(T2I)/len(data))}, Group score: {100*(sum(group_score)/len(data))}, Ipos_2T: {100*(sum(Ipos_2T)/len(data))}, Ineg_2T: {100*(sum(Ineg_2T)/len(data))}, Tpos_2I: {100*(sum(Tpos_2I)/len(data))}, Tneg_2I: {100*(sum(Tneg_2I)/len(data))}')

    df = pd.DataFrame(data)
    df['Ipos_2T'] = [tensor.item() for tensor in Ipos_2T]
    df['Ineg_2T'] = [tensor.item() for tensor in Ineg_2T]
    df['Tpos_2I'] = [tensor.item() for tensor in Tpos_2I]
    df['Tneg_2I'] = [tensor.item() for tensor in Tneg_2I]
    df['I2T'] = [tensor.item() for tensor in I2T]
    df['T2I'] = [tensor.item() for tensor in T2I]
    df['group_score'] = [tensor.item() for tensor in group_score]
    
    df.to_csv(os.path.join(args.save_path, f'BiVLC_{args.run_name}.csv'), index = False, quotechar='"', quoting=csv.QUOTE_ALL)

    df_replace = df[df['type'] == 'replace']
    df_swap = df[df['type'] == 'swap']
    df_add = df[df['type'] == 'add']

    print(f"{args.run_name} checkpoint REPLACE ({len(df_replace)}) scores:   I2T: {100*(sum(df_replace['I2T'])/len(df_replace))}, T2I: {100*(sum(df_replace['T2I'])/len(df_replace))}, Group score: {100*(sum(df_replace['group_score'])/len(df_replace))}, Ipos_2T: {100*(sum(df_replace['Ipos_2T'])/len(df_replace))}, Ineg_2T: {100*(sum(df_replace['Ineg_2T'])/len(df_replace))}, Tpos_2I: {100*(sum(df_replace['Tpos_2I'])/len(df_replace))}, Tneg_2I: {100*(sum(df_replace['Tneg_2I'])/len(df_replace))}")
    print(f"{args.run_name} checkpoint SWAP ({len(df_swap)}) scores:   I2T: {100*(sum(df_swap['I2T'])/len(df_swap))}, T2I: {100*(sum(df_swap['T2I'])/len(df_swap))}, Group score: {100*(sum(df_swap['group_score'])/len(df_swap))}, Ipos_2T: {100*(sum(df_swap['Ipos_2T'])/len(df_swap))}, Ineg_2T: {100*(sum(df_swap['Ineg_2T'])/len(df_swap))}, Tpos_2I: {100*(sum(df_swap['Tpos_2I'])/len(df_swap))}, Tneg_2I: {100*(sum(df_swap['Tneg_2I'])/len(df_swap))}")
    print(f"{args.run_name} checkpoint ADD ({len(df_add)}) scores:   I2T: {100*(sum(df_add['I2T'])/len(df_add))}, T2I: {100*(sum(df_add['T2I'])/len(df_add))}, Group score: {100*(sum(df_add['group_score'])/len(df_add))}, Ipos_2T: {100*(sum(df_add['Ipos_2T'])/len(df_add))}, Ineg_2T: {100*(sum(df_add['Ineg_2T'])/len(df_add))}, Tpos_2I: {100*(sum(df_add['Tpos_2I'])/len(df_add))}, Tneg_2I: {100*(sum(df_add['Tneg_2I'])/len(df_add))}")
