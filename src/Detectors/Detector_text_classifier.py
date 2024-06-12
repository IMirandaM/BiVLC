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
    parser.add_argument('--epoch', type=int,        default=10)
    parser.add_argument('--device',                 default=None)
    parser.add_argument("--seed", type=int,         default=42)
    parser.add_argument('--clip-model',             default="ViT-B-32")
    parser.add_argument('--pretrained',             default="openai")
    parser.add_argument('--model_cache_dir', type=Path,   default='')
    parser.add_argument('--model-checkpoint', type=Path,       default=None)
    parser.add_argument('--dataset',                default='text_classifier')
    parser.add_argument('--save-path', type=Path,   default='./src/Detectors/classifier_ckpt')
    parser.add_argument('--batch-size', type=int,   default=400)
    parser.add_argument('--lr',                     default=1e-6)
    parser.add_argument("--project_name", type=str, default='CLIP_FT_TEXT_CLASSIFIER')
    parser.add_argument("--run_name", type=str,     default='VIT-B-32')

    args = parser.parse_args()

    #os.system('wandb login')

    config = {"model":args.clip_model, "pretrained":args.pretrained,
            "learning_rate": args.lr,
            "epochs": args.epoch, "batch_size": args.batch_size}  

    #wandb.init(project = args.project_name, name = args.run_name, config=config)

    random_seed(seed = args.seed)

    print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    EPOCH = args.epoch
    args.save_path.mkdir(exist_ok=True)

    model, tokenizer, preprocess = load_model(args)
    model.visual = None
    #summary(model, input_size=(args.batch_size, 77))

    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loaded checkpoint from {args.model_checkpoint}')
        
    for param in model.parameters():
        param.requires_grad = False
    
    model = CLIP_text_classifier(model, 1).to(args.device)
    summary(model, input_size=(args.batch_size,77))
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'There are #{trainable_params} trainable parameters of {total_params}.')

    data, collate_fn = dataset(args, preprocess)
  
    train_dataloader = DataLoader(data['train'] , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(data['validation'] , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=False)

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
 
    LAST_EPOCH = 0
    print(f'Fine-tuning for #{EPOCH} epochs.')
    best = 0
    best_epoch = 0
    for epoch in range(LAST_EPOCH, LAST_EPOCH+EPOCH):
        model.train()
        train_losses = []
        train_accuracy = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            label = batch['label'].to(args.device)
            caption =  tokenizer(batch['caption']).to(args.device)
            logits = model(caption)
            #print(logits.shape)
            #print(logits)
            prob = sigmoid(logits)
            #print(prob)
            #print((torch.round(prob)))
            #print(label)
            loss = criterion(prob, label)
            acc =  ((torch.round(prob) == label).sum().float())  / float(label.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Train loss': loss.item(), 'Train acc': acc.item()})
            train_losses.append(loss.item())
            train_accuracy.append(acc.item())
        
        # Validation
        model.eval() 
    
        val_losses = [] 
        val_accuracy = []
        
        with torch.inference_mode():
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                label = batch['label'].to(args.device)
                caption =  tokenizer(batch['caption']).to(args.device)
                logits = model(caption)
                prob = sigmoid(logits)
                val_loss = criterion(prob, label)
                val_acc =  ((torch.round(prob) == label).sum().float()/ float(label.size(0)))
                
                pbar.set_postfix({'Validation loss': val_loss.item(), 'Validation acc': val_acc.item()})
                val_losses.append(val_loss.item())
                val_accuracy.append(val_acc.item())
        #wandb.log({'Train Cross-accuracy': (np.mean(train_accuracy)).item(), 'Train Cross-loss': (np.mean(train_losses)).item(), 'Validation Cross-accuracy' : (np.mean(val_accuracy)).item(), 'Validation Cross-loss': (np.mean(val_losses)).item()})        
        print(f'Epoch {epoch} - Train accuracy: {np.mean(train_accuracy):.4} - Train loss: {np.mean(train_losses):.4} - Validation accuracy: {np.mean(val_accuracy):.4} - Validation loss: {np.mean(val_losses):.4}' )
        if (np.mean(val_accuracy)).item() > best:  #
            try: 
                os.remove(args.save_path / f'best_text_{args.run_name}_{best_epoch}.pt') 
            except OSError:
                pass
            best = (np.mean(val_accuracy)).item()
            best_epoch = epoch
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                },
                args.save_path / f'best_text_{args.run_name}_{epoch}.pt'
            ) 