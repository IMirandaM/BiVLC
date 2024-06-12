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
#import wandb

from src.CLIP_fine_tuning.CLIP_utils import load_model, random_seed, evaluate_SugarCrepe

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',                 default=None)
    parser.add_argument("--seed", type=int,         default=None) #42
    parser.add_argument('--clip-model',             default="ViT-B-32")
    parser.add_argument('--pretrained',             default="openai")
    parser.add_argument('--model_cache_dir', type=Path,   default='')
    parser.add_argument('--model-checkpoint', type=Path,       default=None)
    parser.add_argument('--data-path', type=Path,   default='./source_data/SugarCrepe/test_SugarCrepe.csv')
    parser.add_argument('--img-path', type=Path,    default='/source_data/COCO_2017val2017/')
    parser.add_argument('--save-path', type=Path,   default='./results')
    parser.add_argument("--project_name", type=str, default='CLIP_evaluation_SUGARCREPE')
    parser.add_argument("--run_name", type=str,     default='ViT-B-32')


    args = parser.parse_args()

    #os.system('wandb login')

    config = {"model":args.clip_model, "pretrained":args.pretrained, "fine-tuned":args.run_name}  

    #wandb.init(project = args.project_name, name = args.run_name, config=config)

    if args.seed:
        random_seed(seed = args.seed)
    print(torch.random.initial_seed())

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.save_path.mkdir(exist_ok=True)

    model, tokenizer, preprocess = load_model(args)

    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loaded checkpoint from {args.model_checkpoint}')

    df = pd.read_csv(args.data_path, usecols=['filename','caption','negative_caption','type','subtype'])
    print(f'Evaluating {args.clip_model} fine-tuned in {args.run_name} for {len(df)} instances...')   
    # Validation
    model.eval() 
    correct = []
    with torch.inference_mode():
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            correct.append(evaluate_SugarCrepe(args, row, model, tokenizer, preprocess))
                   
    #wandb.log({'SUGARCREPE-accuracy': np.mean(correct)})        
    print(f'{args.run_name} checkpoint SUGARCREPE OVERALL:  {(np.mean(correct)):.4} ({correct.count(1)}/{len(df)})')
    df['correct'] = correct
    
    df_replace = df[df['type'] == 'replace']
    df_swap = df[df['type'] == 'swap']
    df_add = df[df['type'] == 'add']
    
    print(f"{args.run_name} checkpoint SUGARCREPE REPLACE ({len(df_replace)}):  {(np.mean(df_replace['correct'])):.4}")
    print(f"{args.run_name} checkpoint SUGARCREPE SWAP ({len(df_swap)}):  {(np.mean(df_swap['correct'])):.4}")
    print(f"{args.run_name} checkpoint SUGARCREPE ADD ({len(df_add)}):  {(np.mean(df_add['correct'])):.4}")
    
    df.to_csv(os.path.join(args.save_path, f'SUGARCREPE_CORRECT_{args.run_name}.csv'), index = False, quotechar='"', quoting=csv.QUOTE_ALL)