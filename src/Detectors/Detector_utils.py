from PIL import Image
from torchvision import transforms

import json
import os
import random
import pandas as pd
import numpy as np
import open_clip
import torch
import datasets
from datasets import load_dataset, Dataset

def random_seed(seed=42,):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(args):
    print('Loading model...')
    model, _, transform = open_clip.create_model_and_transforms(
            model_name=args.clip_model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device
        )
    model = model.to(args.device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    print(f'{args.clip_model} loaded with {args.pretrained} ')
    return model, tokenizer, transform

def dataset(args, preprocess):
    print('Loading dataset...')
    if args.dataset == 'image_classifier':
      dataset = load_dataset("imirandam/TROHN-Img")
      print(dataset)
      def collate_fn(batch):
          image_0 = [preprocess(Image.open(os.path.join(args.img_path,images['image_id'])).convert("RGB")) for images in batch] 
          image_1 = [preprocess(images['negative_image'].convert("RGB")) for images in batch] 
          label0 = torch.zeros(len(batch), 1)
          label1 = torch.ones(len(batch), 1)
          image = image_0 + image_1
          image = torch.stack(image)
          #print(len(image))
          label = torch.unbind(label0) + (torch.unbind(label1))
          label = torch.stack(label)

          out = {'label': label,
              'image':image,
              }
          return out
      return dataset, collate_fn
    elif args.dataset == 'text_classifier':
      dataset = load_dataset("imirandam/TROHN-Img")
      print(dataset)
      def collate_fn(batch):
          caption_0 = [captions['caption'] for captions in batch] 
          caption_1 = [captions['negative_caption'] for captions in batch] 
          caption = caption_0 + caption_1
          label0 = torch.zeros(len(batch), 1)
          label1 = torch.ones(len(batch), 1)
          label = torch.unbind(label0) + (torch.unbind(label1))
          label = torch.stack(label)

          out = {'label': label,
              'caption':caption,
              }
          return out
      return dataset, collate_fn
    elif args.dataset == 'text_classifier_eval':
      data = pd.read_csv(args.data_path)
      dataset = Dataset.from_pandas(data)
      print(dataset)
      def collate_fn(batch):
          caption_0 = [captions['caption'] for captions in batch] 
          caption_1 = [captions['negative_caption'] for captions in batch] 
          caption = caption_0 + caption_1
          label0 = torch.zeros(len(batch), 1)
          label1 = torch.ones(len(batch), 1)
          label = torch.unbind(label0) + (torch.unbind(label1))
          label = torch.stack(label)

          out = {'label': label,
              'caption':caption,
              }
          return out
      return dataset, collate_fn
    elif args.dataset == 'classifiers_eval':
      dataset = load_dataset("imirandam/BiVLC", split = "test")
      #print(dataset)
      def collate_fn(batch):
          caption_0 = [captions['caption'] for captions in batch] 
          caption_1 = [captions['negative_caption'] for captions in batch] 
          caption = caption_0 + caption_1
          image_0 = [preprocess(images['image'].convert("RGB")) for images in batch] 
          image_1 = [preprocess(images['negative_image'].convert("RGB")) for images in batch]
          image = image_0 + image_1
          image = torch.stack(image)
          label0 = torch.zeros(len(batch), 1)
          label1 = torch.ones(len(batch), 1)
          label = torch.unbind(label0) + (torch.unbind(label1))
          label = torch.stack(label)

          out = {'label': label,
              'caption':caption,
              'image':image,
              }
          return out
      return dataset, collate_fn
 
