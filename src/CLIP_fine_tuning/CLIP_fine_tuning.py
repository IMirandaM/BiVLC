import json
import os
from pathlib import Path
from PIL import Image

import open_clip
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn.functional import normalize
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
#import wandb

from CLIP_utils import load_model, dataset, random_seed, count_parameters
from scheduler import cosine_lr

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
    parser.add_argument('--dataset',                default='COCO')
    parser.add_argument('--data-path', type=Path,   default='./source_data/COCO_2017/captions_train2017.json')
    parser.add_argument('--img-path', type=Path,    default='./source_data/COCO_2017/train2017/')
    parser.add_argument('--save-path', type=Path,   default='./src/CLIP_fine_tuning/ckpt')
    parser.add_argument('--batch-size', type=int,   default=400)
    parser.add_argument('--lr',                     default=1e-6)
    parser.add_argument('--beta-1',                 default=0.9)
    parser.add_argument('--beta-2',                 default=0.98)
    parser.add_argument('--eps',                    default=1e-6)
    parser.add_argument('--weight-decay',           default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='cosine_lr')
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument("--project_name", type=str, default='CLIP_FT')
    parser.add_argument("--run_name", type=str,     default='VIT-B-32')

    args = parser.parse_args()

    os.system('wandb login')

    config = {"model":args.clip_model, "pretrained":args.pretrained,
            "learning_rate": args.lr, "beta1": args.beta_1, "beta2": args.beta_2, "epsilon": args.eps, "weight-decay": args.weight_decay,
            "epochs": args.epoch, "batch_size": args.batch_size}  

    #wandb.init(project = args.project_name, name = args.run_name, config=config)

    random_seed(seed = args.seed)

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCH = args.epoch
    args.save_path.mkdir(exist_ok=True)

    model, tokenizer, preprocess = load_model(args)
    print(f'There are #{count_parameters(model)} trainable parameters.')

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]


    optimizer = AdamW(
       [{"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.weight_decay}], 
        lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps)
    
    data, collate_fn = dataset(args, preprocess)

    if args.dataset == 'COCO':
        data = data.train_test_split(test_size=0.2, seed=args.seed)
        train_dataloader = DataLoader(data['train'] , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(data['test'] , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=False)
    else:
        train_dataloader = DataLoader(data['train'] , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(data['validation'] , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=False)

    num_batches = len(train_dataloader)
    total_steps = num_batches * args.epoch
    if args.lr_scheduler == "cosine_lr":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup_steps, total_steps)
    
    loss_img = CrossEntropyLoss()
    loss_txt = CrossEntropyLoss()

    LAST_EPOCH = 0
    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loaded checkpoint from {args.model_checkpoint}')

    print(f'Fine-tuning for #{EPOCH} epochs.')
    best = 0
    best_epoch = 0
    for epoch in range(LAST_EPOCH, LAST_EPOCH+EPOCH):
        model.train()
        train_losses = []
        train_accuracy = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        i_accum=0
        for batch in pbar:
            n_images = len(batch['image'])
            step = num_batches * epoch + i_accum
            i_accum += 1
            if args.lr_scheduler == "cosine_lr":
                scheduler(step)
            ground_truth = torch.arange(n_images, dtype=torch.long, device=args.device)
            caption = tokenizer(batch['caption']).to(args.device)
            image =  batch['image'].to(args.device)
            logits_im, logits_text = model.get_logits(image, caption)
            if (args.dataset == 'TROHN-Text'):
                loss = (loss_img(logits_im, ground_truth) + loss_txt(logits_text[:len(logits_im)], ground_truth))/2
                acc_text =  (softmax(logits_text[:len(logits_im)], dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
                acc_im =  (softmax(logits_im, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
            else:
                loss = (loss_img(logits_im, ground_truth) + loss_txt(logits_text, ground_truth))/2
                acc_text =  (softmax(logits_text, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
                acc_im =  (softmax(logits_im, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
            acc = (acc_im + acc_text)/2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Train cross loss': loss.item(), 'Train cross acc': acc.item()})
            train_losses.append(loss.item())
            train_accuracy.append(acc.item())
        
        # Validation
        model.eval() 
    
        val_losses = [] 
        val_accuracy = []
        
        with torch.inference_mode():
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                n_images = len(batch['image'])
                ground_truth = torch.arange(n_images, dtype=torch.long, device=args.device)
                caption = tokenizer(batch['caption']).to(args.device)
                image =  batch['image'].to(args.device)
                logits_im, logits_text = model.get_logits(image, caption)
                if (args.dataset == 'TROHN-Text'):
                    val_loss = (loss_img(logits_im, ground_truth) + loss_txt(logits_text[:len(logits_im)], ground_truth))/2
                    acc_text =  (softmax(logits_text[:len(logits_im)], dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
                    acc_im =  (softmax(logits_im, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
                else:
                    val_loss = (loss_img(logits_im, ground_truth) + loss_txt(logits_text, ground_truth))/2
                    acc_text =  (softmax(logits_text, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
                    acc_im =  (softmax(logits_im, dim=1).argmax(dim=1) == ground_truth).sum().float() / float(ground_truth.size(0))
                val_acc = (acc_im + acc_text)/2
                pbar.set_postfix({'Validation cross loss': val_loss.item(), 'Validation cross acc': val_acc.item()})
                val_losses.append(val_loss.item())
                val_accuracy.append(val_acc.item())
        #wandb.log({'Train Cross-accuracy': (np.mean(train_accuracy)).item(), 'Train Cross-loss': (np.mean(train_losses)).item(), 'Validation Cross-accuracy' : (np.mean(val_accuracy)).item(), 'Validation Cross-loss': (np.mean(val_losses)).item()})        
        print(f'Epoch {epoch} - Train Cross-accuracy: {np.mean(train_accuracy):.4} - Train Cross-loss: {np.mean(train_losses):.4} - Validation Cross-accuracy: {np.mean(val_accuracy):.4} - Validation Cross-loss: {np.mean(val_losses):.4}' )
        if (np.mean(val_accuracy)).item() > best:  #
            try: 
                os.remove(args.save_path / f'best_{args.run_name}_{best_epoch}.pt') 
            except OSError:
                pass
            best = (np.mean(val_accuracy)).item()
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                },
                args.save_path / f'best_{args.run_name}_{epoch}.pt'
            )
            best_epoch = epoch 