import pandas as pd
import numpy as np
import csv
import os

from adversarial_refine import adversarial_refine

treshold = 0.645 # top 35 percentile, we seek to obtain a number of image-text pairs similar to COCO 2017 train.

subtypes = ['REPLACE_OBJ','REPLACE_ATT', 'REPLACE_REL' ,'SWAP_OBJ','SWAP_ATT','ADD_OBJ', 'ADD_ATT']
for subtype in subtypes:
    keep = []
    df = pd.read_csv(f'./src/Training_data/TROHN_Img/data/TROHN-TEXT_{subtype}_SCORES.csv')
    df = df[df['Similarity'] < 0.999] # Captions and generated negative captions with a similarity greater than 0.999 are the same, so they are removed. You can check this in the scoring documents.
    df_vera = df.NEG_VERA.quantile(treshold)
    df_cola = df.NEG_CoLA.quantile(treshold)
    globals() [subtype] = df[(df['NEG_VERA'] >= df_vera) & (df['NEG_CoLA'] >=  df_cola)]


sub_df = [REPLACE_OBJ,REPLACE_ATT, REPLACE_REL ,SWAP_OBJ,SWAP_ATT,ADD_OBJ, ADD_ATT]
print([f'{str(x)}: {len(y)} ' for x,y in zip(subtypes,sub_df)])
data = pd.concat(sub_df).reset_index( drop = True)
# Removing questions or not questions without final dot
data.loc[:,"negative_caption"] = [x if x[-1] == "." else x[-1] for x in data["negative_caption"]]
data = data.loc[data["negative_caption"].str.len() > 10]

data["prompt"] = data["negative_caption"].str.replace(".","") + ", high resolution, professional, 4k, highly detailed"
data = data.drop_duplicates(subset=['prompt'])
print(f'Total before refinement: {len(data)}')

keep = adversarial_refine(list(data["PAIR_VERA"]),list(data["PAIR_CoLA"]))
selected = data.iloc[keep]
print(f'Total: {len(selected)}')

selected['caption'] = selected['caption'].apply(lambda x: (x.strip().capitalize()) +'.' if x[-1] != '.' else x.strip().capitalize())
selected['negative_caption'] = selected['negative_caption'].apply(lambda x: (x.strip().capitalize()) +'.' if x[-1] != '.' else x.strip().capitalize())

selected.to_csv(f'./src/Training_data/TROHN_Img/data/TROHN-Img_generation.csv', columns = ["image_id","caption","negative_caption","prompt"], index = False, quotechar='"', quoting=csv.QUOTE_ALL)