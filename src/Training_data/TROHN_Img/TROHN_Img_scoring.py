import os
import pandas as pd
from pathlib import Path
import csv
from tqdm import tqdm
import argparse
import transformers
import torch
import open_clip
from torch.nn.functional import softmax, normalize
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

class similarity:
    def __init__(self, model, pretrained):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=args.device)
        self.model = self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)

    def run(self, cap, neg_cap):
        with torch.no_grad():
            t_cap = self.tokenizer(cap, context_length=77)
            t_neg_cap = self.tokenizer(neg_cap, context_length=77)
        
            # TEXT
            t_cap_emb = self.model.encode_text(t_cap.to(args.device), normalize=True)
            t_neg_cap_emb = self.model.encode_text(t_neg_cap.to(args.device), normalize=True)
            text_similarity = t_cap_emb.cpu().numpy() @ t_neg_cap_emb.T.cpu().numpy()
        return text_similarity.item()

class VERA: # CODE FROM https://huggingface.co/spaces/liujch1998/vera/blob/main/app.py#L27-L98
    def __init__(self, model):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)#, cache_dir="/gaueko0/users/imiranda014/.cache/huggingface/hub"
        self.model = transformers.T5EncoderModel.from_pretrained(model, low_cpu_mem_usage=True, device_map='auto', torch_dtype='auto', offload_folder='offload')#, cache_dir="/gaueko0/users/imiranda014/.cache/huggingface/hub"
        self.model.D = self.model.shared.embedding_dim
        self.linear = torch.nn.Linear(self.model.D, 1, dtype=self.model.dtype).to(args.device)
        self.linear.weight = torch.nn.Parameter(self.model.shared.weight[32099, :].unsqueeze(0)) # (1, D)
        self.linear.bias = torch.nn.Parameter(self.model.shared.weight[32098, 0].unsqueeze(0)) # (1)
        self.model.eval()
        self.t = self.model.shared.weight[32097, 0].item()

    def run(self, statement):
        input_ids = self.tokenizer.batch_encode_plus([statement], return_tensors='pt', padding='longest', truncation='longest_first', max_length=128).input_ids.to(args.device)
        with torch.no_grad():
            output = self.model(input_ids)
            last_hidden_state = output.last_hidden_state.to(args.device) # (B=1, L, D)
            hidden = last_hidden_state[0, -1, :] # (D)
            logit = self.linear(hidden).squeeze(-1) # ()
            logit_calibrated = logit / self.t
            # score = logit.sigmoid()
            score_calibrated = logit_calibrated.sigmoid()
        return score_calibrated.item()

    def runs(self, statements):
        tok = self.tokenizer.batch_encode_plus(statements, return_tensors='pt', padding='longest')
        input_ids = tok.input_ids.to(args.device)
        attention_mask = tok.attention_mask.to(args.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_indices = attention_mask.sum(dim=1, keepdim=True) - 1 # (B, 1)
            last_indices = last_indices.unsqueeze(-1).expand(-1, -1, self.model.D) # (B, 1, D)
            last_hidden_state = output.last_hidden_state.to(args.device) # (B, L, D)
            hidden = last_hidden_state.gather(dim=1, index=last_indices).squeeze(1) # (B, D)
            logits = self.linear(hidden).squeeze(-1) # (B)
            logits_calibrated = logits / self.t
            scores = logits.sigmoid()
            scores_calibrated = logits_calibrated.sigmoid()
        return [score for score in scores_calibrated]


class CoLA:
    def __init__(self, model):
        #self.pipe = pipeline("text-classification", model="textattack/roberta-base-CoLA")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, device_map = 'auto')

    def run(self, statement):
        with torch.inference_mode():
            encoded_input = self.tokenizer(statement, return_tensors='pt').to(args.device)
            output = self.model(**encoded_input)
            scores = output[0][0].detach().cpu()
            scores = softmax(scores)
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '--names-list', nargs='+', default=['all'])
    parser.add_argument('--data-path', type=Path,   default='./src/Training_data/TROHN_Text/data')
    parser.add_argument('--save-path', type=Path,   default='./src/Training_data/TROHN_Img/data')
    parser.add_argument('--device', default=None)
  
    args = parser.parse_args()
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    list_types = ['REPLACE_OBJ','REPLACE_ATT','REPLACE_REL', 'SWAP_OBJ','SWAP_ATT','ADD_OBJ', 'ADD_ATT'] 
    
    if args.type[0] == 'all':
        subtypes = list_types 
    else:
        subtypes = args.type  # 'REPLACE_OBJ','REPLACE_ATT','REPLACE_REL', 'SWAP_OBJ','SWAP_ATT','ADD_OBJ', 'ADD_ATT'
    
    model = similarity('ViT-B-32', 'openai')
    model1 = VERA('liujch1998/vera')
    model2 = CoLA('textattack/roberta-base-CoLA')

    for subtype in subtypes:
        clean_data = pd.read_csv(os.path.join(args.data_path, f'TROHN-TEXT_{subtype}.csv')) #, nrows=5
        similarity_scores = []
        pos_vera_scores = []
        neg_vera_scores = []
        pos_cola_scores = []
        neg_cola_scores = []
        for ind in tqdm(clean_data.index):
            similarity_score = similarity.run(model, clean_data['caption'][ind], clean_data['negative_caption'][ind])
            similarity_scores.append(similarity_score)
            # VERA for positive and negative
            pos_vera_score = VERA.run(model1, clean_data['caption'][ind])
            pos_vera_scores.append(pos_vera_score)
            neg_vera_score = VERA.run(model1, clean_data['negative_caption'][ind])
            neg_vera_scores.append(neg_vera_score)
            # COLA for positive and negative
            pos_cola_score = CoLA.run(model2, clean_data['caption'][ind])
            pos_cola_scores.append(pos_cola_score[1].item())
            neg_cola_score = CoLA.run(model2, clean_data['negative_caption'][ind])
            neg_cola_scores.append(neg_cola_score[1].item())
        clean_data['Similarity'] = similarity_scores
        clean_data['POS_VERA'] = pos_vera_scores
        clean_data['NEG_VERA'] = neg_vera_scores
        clean_data['POS_CoLA'] = pos_cola_scores
        clean_data['NEG_CoLA'] = neg_cola_scores
        clean_data['PAIR_VERA'] = clean_data['POS_VERA']  - clean_data['NEG_VERA']
        clean_data['PAIR_CoLA'] = clean_data['POS_CoLA'] - clean_data['NEG_CoLA']
        clean_data.to_csv(os.path.join(args.save_path, f'TROHN-TEXT_{subtype}_SCORES.csv'), index = False, quotechar='"', quoting=csv.QUOTE_ALL)