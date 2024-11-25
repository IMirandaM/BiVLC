import os
from PIL import Image
from tqdm import tqdm
import clip
from datasets import load_dataset
import t2v_metrics
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="clip-flant5-xxl") #'clip-flant5-xxl' 'clip-flant5-xl'
    args = parser.parse_args()

    img_path = '/gaueko0/users/imiranda014/GSCRATCH/Data/COCO/val2017'

    print('Loading data...')
    data = pd.read_csv('/gaueko0/users/imiranda014/Esperimentuak/SugarCrepe/t2v_metrics/test_SugarCrepe.csv')
    print('Data loaded!')

    model = args.model 
    print(f'Loading model {model}...')
    clip_flant5_score = t2v_metrics.VQAScore(model=model) 
    print('Model loaded!')

    metric = []
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        images = [os.path.join(img_path, row['filename'])]
        texts = [row['caption'], row['negative_caption']]
        scores = clip_flant5_score(images=images, texts=texts) # scores[i][j] is the score between image i and text j

        C0_I0 = scores[0][0]
        C1_I0 = scores[0][1]

        score = C0_I0 > C1_I0

        metric.append(score)

    data['score'] = [tensor.item() for tensor in metric]

    data_replace = data[data['type'] == 'replace']
    data_swap = data[data['type'] == 'swap']
    data_add = data[data['type'] == 'add']

    print(f"VQAScore checkpoint OVERALL ({len(data)}) scores: {100*(sum(data['score'])/len(data)):.4}")
    print(f"VQAScore checkpoint REPLACE ({len(data_replace)}) scores: {100*(sum(data_replace['score'])/len(data_replace)):.4}")
    print(f"VQAScore checkpoint SWAP ({len(data_swap)}) scores: {100*(sum(data_swap['score'])/len(data_swap)):.4}")
    print(f"VQAScore checkpoint ADD ({len(data_add)}) scores: {100*(sum(data_add['score'])/len(data_add)):.4}")