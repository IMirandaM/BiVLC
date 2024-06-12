import json
import os
from pathlib import Path
from PIL import Image

import open_clip
import torch
from torchsummary import summary
from torch.nn import BCELoss
from torch.nn.functional import softmax, sigmoid
from torch.optim import Adam, AdamW
from tqdm import tqdm
from torch.nn.functional import normalize
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import csv
#import wandb

from Detector_utils import load_model, dataset, random_seed

class CLIP_text_classifier(torch.nn.Module):
    def __init__(self, encoder_model, n_output):
        super().__init__()
        
        self.text_encoder = encoder_model
        self.classifier = torch.nn.Linear(in_features=512, 
                    out_features=n_output)
        
    def forward(self, x):
        x = self.text_encoder.encode_text(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',                 default=None)
    parser.add_argument("--seed", type=int,         default=42)
    parser.add_argument('--clip-model',             default="ViT-B-32")
    parser.add_argument('--pretrained',             default="openai")
    parser.add_argument('--dataset',                default='text_classifier_eval')
    parser.add_argument('--model_cache_dir', type=Path,   default='')
    parser.add_argument('--text-checkpoint', type=Path,       default=None)
    parser.add_argument('--data-path', type=Path,   default='./source_data/SugarCrepe/test_SugarCrepe.csv')
    parser.add_argument('--batch-size', type=int,   default=1000)
    parser.add_argument('--save-path', type=Path,   default='./src/Detectors/results')
    parser.add_argument("--project_name", type=str, default='CLIP_evaluation_SugarCrepe_classifier') 
    parser.add_argument("--run_name", type=str,     default='ViT-B-32')

    args = parser.parse_args()

    #os.system('wandb login')

    config = {"model":args.clip_model, "pretrained":args.pretrained, "batch_size": args.batch_size}  

    #wandb.init(project = args.project_name, name = args.run_name, config=config)

    random_seed(seed = args.seed)

    print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, preprocess = load_model(args)
    model.visual = None
    #summary(model, input_size=(args.batch_size, 77))

    model = CLIP_text_classifier(model, 1).to(args.device)
    summary(model, input_size=(args.batch_size, 77))

    if args.text_checkpoint:
        checkpoint = torch.load(args.text_checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loaded checkpoint from {args.text_checkpoint}')
        
    data, collate_fn = dataset(args, preprocess)
 
    test_dataloader = DataLoader(data , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=False)

    print(f'Evaluating {args.clip_model} fine-tuned in {args.run_name} for {len(data)} instances...')   
    # Validation
    model.eval() 
    accuracy = []
    pred = []
    l = []
    with torch.inference_mode():
        pbar = tqdm(test_dataloader)
        for batch in pbar:
            label = batch['label'].to(args.device)
            caption =  tokenizer(batch['caption']).to(args.device)
            logits = model(caption)
            prob = sigmoid(logits)
            pred.extend(prob) #torch.round(prob).tolist()
            l.extend(label.tolist())
            #print((torch.round(prob) == label).sum())
            #print(float(label.size(0)))
            acc =  ((torch.round(prob) == label).sum().float())  / float(label.size(0))
            #print(acc)
            accuracy.append(acc.item())
       
    #wandb.log({'Accuracy': np.mean(accuracy).item()})
    pred = [float(item) for inner_list in pred for item in inner_list] 
    l = [float(item) for inner_list in l for item in inner_list] 
    dict = {'text_pred': pred, 'label': l}
    df = pd.DataFrame(dict)

    print(f'{args.run_name} checkpoint pair accuracy:  {(np.mean(accuracy)).item()}')

    positives = df[df['label'] == 0]
    negatives = df[df['label'] == 1]

    text_score_i0 = []

    for c0, c1 in zip(positives['text_pred'], negatives['text_pred']):
        if (c0 < c1):
            text_score_i0.append(1)
        else:
            text_score_i0.append(0)
    print(f'Text_score_i0: {100*np.mean(text_score_i0)}')

    df.to_csv(os.path.join(args.save_path, f'SUGARCREPE_Classifier_{args.run_name}.csv'), index = False, quotechar='"', quoting=csv.QUOTE_ALL)  