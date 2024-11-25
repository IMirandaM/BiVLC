import os
from PIL import Image
from tqdm import tqdm
import clip
from datasets import load_dataset
import t2v_metrics
import pandas as pd

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="clip-flant5-xxl") #'clip-flant5-xxl' 'clip-flant5-xl'
    args = parser.parse_args()
    
    img_path = '/gaueko0/users/imiranda014/Esperimentuak/SugarCrepe/t2v_metrics/img'

    print('Loading data...')
    data = load_dataset("imirandam/BiVLC", split = "test")
    print('Data loaded!')

    model = args.model 
    print(f'Loading model {model}...')
    clip_flant5_score = t2v_metrics.VQAScore(model=model) 
    print('Model loaded!')

    iter=data.iter(batch_size=1)
    I2T = []
    T2I = []
    group_score = []

    Ipos_2T = []
    Ineg_2T = []
    Tpos_2I = []
    Tneg_2I = []

    image = os.path.join(img_path, "image.png")
    negative_image = os.path.join(img_path, "negative_image.png")

    for item in tqdm(iter):
      item['image'][0].save(image)
      item['negative_image'][0].save(negative_image)
      images = [image, negative_image]
      texts = [item['caption'][0], item['negative_caption'][0]]
      scores = clip_flant5_score(images=images, texts=texts) # scores[i][j] is the score between image i and text j

      C0_I0 = scores[0][0]
      C1_I0 = scores[0][1]
      C0_I1 = scores[1][0]
      C1_I1 = scores[1][1]

      Ipos_2T_s = C0_I0 > C1_I0
      Ineg_2T_s = C1_I1 > C0_I1
      Tpos_2I_s = C0_I0 > C0_I1
      Tneg_2I_s = C1_I1 > C1_I0

      I2T_s = Ipos_2T_s and Ineg_2T_s
      T2I_s = Tpos_2I_s and Tneg_2I_s
      group_s = I2T_s and T2I_s

      I2T.append(I2T_s)
      T2I.append(T2I_s)
      group_score.append(group_s)
      Ipos_2T.append(Ipos_2T_s)
      Ineg_2T.append(Ineg_2T_s)
      Tpos_2I.append(Tpos_2I_s)
      Tneg_2I.append(Tneg_2I_s)

    print(f'VQAScore checkpoint OVERALL scores:   I2T: {100*(sum(I2T)/len(data)):.4}, T2I: {100*(sum(T2I)/len(data)):.4}, Group score: {100*(sum(group_score)/len(data)):.4}, Ipos_2T: {100*(sum(Ipos_2T)/len(data)):.4}, Ineg_2T: {100*(sum(Ineg_2T)/len(data)):.4}, Tpos_2I: {100*(sum(Tpos_2I)/len(data)):.4}, Tneg_2I: {100*(sum(Tneg_2I)/len(data)):.4}')

    df = pd.DataFrame(data)
    df['Ipos_2T'] = [tensor.item() for tensor in Ipos_2T]
    df['Ineg_2T'] = [tensor.item() for tensor in Ineg_2T]
    df['Tpos_2I'] = [tensor.item() for tensor in Tpos_2I]
    df['Tneg_2I'] = [tensor.item() for tensor in Tneg_2I]
    df['I2T'] = [tensor.item() for tensor in I2T]
    df['T2I'] = [tensor.item() for tensor in T2I]
    df['group_score'] = [tensor.item() for tensor in group_score]

    df_replace = df[df['type'] == 'replace']
    df_swap = df[df['type'] == 'swap']
    df_add = df[df['type'] == 'add']

    print(f"VQAScore checkpoint REPLACE ({len(df_replace)}) scores:   I2T: {100*(sum(df_replace['I2T'])/len(df_replace)):.4}, T2I: {100*(sum(df_replace['T2I'])/len(df_replace)):.4}, Group score: {100*(sum(df_replace['group_score'])/len(df_replace)):.4}, Ipos_2T: {100*(sum(df_replace['Ipos_2T'])/len(df_replace)):.4}, Ineg_2T: {100*(sum(df_replace['Ineg_2T'])/len(df_replace)):.4}, Tpos_2I: {100*(sum(df_replace['Tpos_2I'])/len(df_replace)):.4}, Tneg_2I: {100*(sum(df_replace['Tneg_2I'])/len(df_replace)):.4}")
    print(f"VQAScore checkpoint SWAP ({len(df_swap)}) scores:   I2T: {100*(sum(df_swap['I2T'])/len(df_swap)):.4}, T2I: {100*(sum(df_swap['T2I'])/len(df_swap)):.4}, Group score: {100*(sum(df_swap['group_score'])/len(df_swap)):.4}, Ipos_2T: {100*(sum(df_swap['Ipos_2T'])/len(df_swap)):.4}, Ineg_2T: {100*(sum(df_swap['Ineg_2T'])/len(df_swap)):.4}, Tpos_2I: {100*(sum(df_swap['Tpos_2I'])/len(df_swap)):.4}, Tneg_2I: {100*(sum(df_swap['Tneg_2I'])/len(df_swap)):.4}")
    print(f"VQAScore checkpoint ADD ({len(df_add)}) scores:   I2T: {100*(sum(df_add['I2T'])/len(df_add)):.4}, T2I: {100*(sum(df_add['T2I'])/len(df_add)):.4}, Group score: {100*(sum(df_add['group_score'])/len(df_add)):.4}, Ipos_2T: {100*(sum(df_add['Ipos_2T'])/len(df_add)):.4}, Ineg_2T: {100*(sum(df_add['Ineg_2T'])/len(df_add)):.4}, Tpos_2I: {100*(sum(df_add['Tpos_2I'])/len(df_add)):.4}, Tneg_2I: {100*(sum(df_add['Tneg_2I'])/len(df_add)):.4}")