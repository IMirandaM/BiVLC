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
    if args.dataset == 'COCO':
      with open(args.data_path, 'r') as f:
        data = json.load(f)
        data = data['annotations']

      img_cap_pairs = []

      for sample in data:
          img_name = '%012d.jpg' % sample['image_id']
          img_cap_pairs.append([img_name, sample['caption']])

      captions = pd.DataFrame(img_cap_pairs, columns=['image_id', 'caption'])
      captions['caption'] = captions['caption'].str.strip()
      captions['caption'] = captions['caption'].str.replace('\n', ' ')
      captions = captions.reset_index(drop=True)
      data = captions
      dataset = Dataset.from_pandas(data)
      print(dataset)
      def collate_fn(batch):
          id = [images['image_id'] for images in batch] 
          image = [preprocess(Image.open(os.path.join(args.img_path,images['image_id'])).convert("RGB")) for images in batch] 
          image = torch.stack(image)
          caption = [captions['caption'] for captions in batch] 

          out = {'img_id': id,
              'caption': caption,
              'image':image,
              }
          return out
      return dataset, collate_fn 
    elif args.dataset == 'TROHN-Text':
      dataset = load_dataset("imirandam/TROHN-Text")
      print(dataset)
      def collate_fn(batch):
        id = [images['image_id'] for images in batch] 
        image = [preprocess(Image.open(os.path.join(args.img_path,images['image_id'])).convert("RGB")) for images in batch] 
        image = torch.stack(image)
        positive = [captions['caption'] for captions in batch]
        negative = [captions['negative_caption'] for captions in batch] 
        caption = positive + negative

        out = {'img_id': id,
              'caption': caption,
              'image':image,
              }
        return out
      return dataset, collate_fn
    elif args.dataset == 'TROHN-Img':
      dataset = load_dataset("imirandam/TROHN-Img")
      print(dataset)
      def collate_fn(batch):
          image_0 = [preprocess(Image.open(os.path.join(args.img_path,images['image_id'])).convert("RGB")) for images in batch] 
          image_1 = [preprocess(images['negative_image'].convert("RGB")) for images in batch] 
          caption_0 = [captions['caption'] for captions in batch] 
          caption_1 = [captions['negative_caption'] for captions in batch] 
          image = image_0 + image_1
          image = torch.stack(image)
          caption = caption_0 + caption_1

          out = {'img_id': id,
              'caption': caption,
              'image':image,
              }
          return out
      return dataset, collate_fn
    elif args.dataset == 'evaluation':
      dataset = load_dataset("imirandam/BiVLC", split = "test")
      print(len(dataset))
      def collate_fn(batch):
          image_0 = [preprocess(images['image'].convert("RGB")) for images in batch] 
          image_0 = torch.stack(image_0)
          image_1 = [preprocess(images['negative_image'].convert("RGB")) for images in batch] 
          image_1 = torch.stack(image_1)
          caption_0 = [captions['caption'] for captions in batch] 
          caption_1 = [captions['negative_caption'] for captions in batch] 

          out = {'caption_0': caption_0,
                 'caption_1': caption_1,
              'image_0':image_0,
              'image_1':image_1,
              }
          return out
      return dataset, collate_fn
    

def evaluate_SugarCrepe(args, row, model, tokenizer, preprocess):
  """"Evaluation based on SUGARCREPE repository code""" 
  with torch.inference_mode():
    pos_text = tokenizer(row['caption']).to(args.device)
    pos_text_embedding = model.encode_text(pos_text, normalize=True)
    neg_text = tokenizer(row['negative_caption']).to(args.device)
    neg_text_embedding = model.encode_text(neg_text, normalize=True)
    image = preprocess(Image.open(os.path.join(args.img_path,row['filename'])))
    image_embedding = model.encode_image(image.unsqueeze(dim=0).to(args.device), normalize=True)
    pos_score = pos_text_embedding @ image_embedding.t()
    neg_score = neg_text_embedding @ image_embedding.t()
    return 1 if pos_score.item() > neg_score.item() else 0

 
def evaluate_BiVLC(args, batch, model, tokenizer):
    image_0_embedding = model.encode_image(batch["image_0"].to(args.device), normalize=True)
    image_1_embedding = model.encode_image(batch["image_1"].to(args.device), normalize=True)
    caption_0_embedding = model.encode_text(tokenizer(batch['caption_0']).to(args.device), normalize=True)
    caption_1_embedding = model.encode_text(tokenizer(batch['caption_1']).to(args.device), normalize=True)

    sim_C0_I0 = caption_0_embedding @ image_0_embedding.t()
    sim_C0_I1 = caption_0_embedding @ image_1_embedding.t()
    sim_C1_I0 = caption_1_embedding @ image_0_embedding.t()
    sim_C1_I1 = caption_1_embedding @ image_1_embedding.t()

    sim_C0_I0 = torch.diagonal(sim_C0_I0)
    sim_C0_I1 = torch.diagonal(sim_C0_I1)
    sim_C1_I0 = torch.diagonal(sim_C1_I0)
    sim_C1_I1 = torch.diagonal(sim_C1_I1)

    l_I2T = []
    l_T2I = []
    l_group_score = []
    l_Ipos_2T = []
    l_Ineg_2T = []
    l_Tpos_2I = []
    l_Tneg_2I = []

    for s_C0_I0,s_C0_I1, s_C1_I0, s_C1_I1 in zip(sim_C0_I0,sim_C0_I1, sim_C1_I0, sim_C1_I1):
      Ipos_2T = s_C0_I0 > s_C1_I0
      Ineg_2T = s_C1_I1 > s_C0_I1
      Tpos_2I = s_C0_I0 > s_C0_I1 
      Tneg_2I = s_C1_I1 > s_C1_I0

      I2T = Ipos_2T and Ineg_2T
      T2I = Tpos_2I and Tneg_2I
      group_score = I2T and T2I

      l_I2T.append(I2T)
      l_T2I.append(T2I)
      l_group_score.append(group_score)

      l_Ipos_2T.append(Ipos_2T)
      l_Ineg_2T.append(Ineg_2T)
      l_Tpos_2I.append(Tpos_2I)
      l_Tneg_2I.append(Tneg_2I)

    return [l_I2T, l_T2I, l_group_score, l_Ipos_2T, l_Ineg_2T, l_Tpos_2I, l_Tneg_2I]
