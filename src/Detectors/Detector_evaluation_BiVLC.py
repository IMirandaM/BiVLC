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

class CLIP_image_classifier(torch.nn.Module):
    def __init__(self, encoder_model, n_output):
        super().__init__()
        
        self.image_encoder = list(encoder_model.children())[0]
        self.classifier = torch.nn.Linear(in_features=512, 
                    out_features=n_output,
                    bias=True)
        
    def forward(self, x):
        x = self.image_encoder(x)
        x = self.classifier(x)
        return x
    
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
    parser.add_argument('--dataset',                default='classifiers_eval')
    parser.add_argument('--model_cache_dir', type=Path,   default='')
    parser.add_argument('--image-checkpoint', type=Path,       default=None)
    parser.add_argument('--text-checkpoint', type=Path,       default=None)
    parser.add_argument('--batch-size', type=int,   default=1000)
    parser.add_argument('--save-path', type=Path,   default='./src/Detectors/results')
    parser.add_argument("--project_name", type=str, default='CLIP_evaluation_BiVLC_classifier')
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
    #summary(model, input_size=(args.batch_size, 77))

    image_classifier = CLIP_image_classifier(model, 1).to(args.device)
    model.visual = None
    text_classifier = CLIP_text_classifier(model, 1).to(args.device)
    print('Image classifier')
    summary(image_classifier, input_size=(args.batch_size, 3, 224, 224))
    print('Text classifier')
    summary(text_classifier, input_size=(args.batch_size, 77))
    
    if args.image_checkpoint:
        checkpoint = torch.load(args.image_checkpoint)
        if 'state_dict' in checkpoint:
            image_classifier.load_state_dict(checkpoint['state_dict'])
        else:
            image_classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f'image_classifier loaded checkpoint from {args.image_checkpoint}')

    if args.text_checkpoint:
        checkpoint = torch.load(args.text_checkpoint)
        if 'state_dict' in checkpoint:
            text_classifier.load_state_dict(checkpoint['state_dict'])
        else:
            text_classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f'text_classifier loaded checkpoint from {args.text_checkpoint}')
        
    data, collate_fn = dataset(args, preprocess)
 
    test_dataloader = DataLoader(data , collate_fn = collate_fn, batch_size=args.batch_size, shuffle=False)

    print(data)
    print(f'Evaluating {args.clip_model} fine-tuned in {args.run_name} for {len(data)} instances...')   
    # Validation
    image_classifier.eval()
    text_classifier.eval()
    accuracy = []
    text_list = []
    img_list = []
    label_list = []
    text_acc = []
    img_acc = []
    with torch.inference_mode():
        pbar = tqdm(test_dataloader)
        for batch in pbar:
            label = batch['label'].to(args.device)
            caption =  tokenizer(batch['caption']).to(args.device)
            text_logits = text_classifier(caption)
            text_prob = sigmoid(text_logits)
            image =  batch['image'].to(args.device)
            img_logits = image_classifier(image)
            img_prob = sigmoid(img_logits)
            for text, img, lab in zip(text_prob, img_prob, label): # torch.round()
                #print(text, img, lab)
                text_list.append(text.item())
                img_list.append(img.item())
                label_list.append(lab.item())
                if (torch.round(text) == lab):
                    text_acc.append(1)
                else:
                    text_acc.append(0)
                
                if (torch.round(img) == lab):
                    img_acc.append(1)
                else:
                    img_acc.append(0)
    
    dict = {'text_pred': text_list, 'img_pred': img_list, 'label': label_list}
    df = pd.DataFrame(dict)

    print(f'{args.run_name} checkpoint text accuracy:  {np.mean(text_acc)}')
    print(f'{args.run_name} checkpoint img accuracy:  {np.mean(img_acc)}')

    positives = df[df['label'] == 0]
    negatives = df[df['label'] == 1]

    text_score_i0 = []
    text_score_i1 = []
    img_score_c0 = []
    img_score_c1 = []
    text_score = []
    img_score = []
    group_score = []

    for c0, i0, c1 , i1 in zip(positives['text_pred'], positives['img_pred'], negatives['text_pred'], negatives['img_pred']):
        t_s_i0 = False
        t_s_i1 = False
        i_s_c0 = False
        i_s_c1 = False
        t_s = False
        i_s = False

        if (np.round(i0) == 0) & (c0 < c1):
            t_s_i0 = True
            text_score_i0.append(1)
        else:
            text_score_i0.append(0)
        if (np.round(i1) == 1) & (c0 < c1):
            t_s_i1 = True
            text_score_i1.append(1)
        else:
            text_score_i1.append(0)
        if (np.round(c0) == 0) & (i0 < i1):
            i_s_c0 = True
            img_score_c0.append(1)
        else:
            img_score_c0.append(0)
        if (np.round(c1) == 1) & (i0 < i1):
            i_s_c1 = True
            img_score_c1.append(1)
        else:
            img_score_c1.append(0)
        # Performance
        if (t_s_i0) & (t_s_i1):
            t_s = True
            text_score.append(1)
        else:
            text_score.append(0)
        if (i_s_c0) & (i_s_c1):
            i_s = True
            img_score.append(1)
        else:
            img_score.append(0)
        if (t_s) & (i_s):
            group_score.append(1)
        else:
            group_score.append(0)

    print(f'Text_score: {100*np.mean(text_score)}, Image_score: {100*np.mean(img_score)}, Group_score: {100*np.mean(group_score)}')
    print(f'Text_score_i0: {100*np.mean(text_score_i0)}, Text_score_i1: {100*np.mean(text_score_i1)}, Image_score_c0: {100*np.mean(img_score_c0)}, Image_score_c1: {100*np.mean(img_score_c1)}')

   
    df.to_csv(os.path.join(args.save_path, f'BiVLC_Classifier_CORRECT_{args.run_name}.csv'), index = False, quotechar='"', quoting=csv.QUOTE_ALL) #UNBIASED_CAPITALIZE_

    #wandb.log({'Accuracy': np.mean(accuracy)})        