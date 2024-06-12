import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re
import numpy as np
import csv
import transformers
import torch
import argparse
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '--names-list', nargs='+', default=['all'])
    parser.add_argument('--data-path', type=Path,   default='./source_data/COCO_2017/captions_train2017.json')
    parser.add_argument('--save-path', type=Path,   default='./src/Training_data/TROHN_Text/data')
    parser.add_argument('--device', default=None)
  
    args = parser.parse_args()
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"


    with open(args.data_path, 'r') as f:
        data = json.load(f)
        data = data['annotations']

    img_cap_pairs = []

    for sample in data:
        img_name = '%012d.jpg' % sample['image_id']
        img_cap_pairs.append([img_name, sample['caption']])

    captions = pd.DataFrame(img_cap_pairs, columns=['image_id', 'caption'])
    captions = captions[:10]
    captions['caption'] = captions['caption'].str.strip()
    captions['caption'] = captions['caption'].str.replace('\n', ' ')
    captions = captions.reset_index(drop=True)
    print(f'Number of unique images {len(captions["image_id"].unique())}')
    print(f'Number of captions {len(captions)}')

    source_data = captions
    print(f'You have loaded data with {len(source_data)} instances from {args.data_path}')

    pipe = pipeline("text-generation", model="openchat/openchat-3.5-0106", torch_dtype=torch.bfloat16, device_map="auto")

    templates = {'REPLACE_OBJ':[
                        {"role": "user", "content": """Given an input sentence describing a scene, your task is to:
                            1. Locate the noun words in the sentence.
                            2. Randomly pick one noun word.
                            3. Replace the selected noun word with a new noun word to make a new sentence.
                        The new sentence must meet the following three requirements:
                            1. The new sentence must be describing a scene that is as different as possible from the original scene.
                            2. The new sentence must be fluent and grammatically correct.
                            3. The new sentence must make logical sense.
                        Here are some examples:
                        Original sentence: A man is in a kitchen making pizzas.
                        Nouns: ["man", "kitchen", "pizzas"]
                        Selected noun: man
                        New noun: woman
                        New sentence: A woman is in a kitchen making pizzas.\n
                        Original sentence: a woman seated on wall and birds besides her"""},
                {"role": "assistant", "content": """Nouns: ['woman', 'wall', 'birds']
                            Selected noun: wall
                            New noun: bench
                            New sentence: A woman seated on a bench and birds besides her.\n"""}],
                'REPLACE_ATT':
                        [{"role": "user", "content": """Given an input sentence describing a scene, your task is to:
                            1. Locate the adjective words describing objects in the sentence. If there are no adjective words, return an empty list.
                            2. Randomly pick one adjective word.
                            3. Replace the selected adjective word with a new adjective word to make a new sentence.
                        The new sentence must meet the following three requirements:
                            1. The new sentence must be describing a scene that is as different as possible from the original scene.
                            2. The new sentence must be fluent and grammatically correct.
                            3. The new sentence must make logical sense.
                        Here are some examples:
                        Original sentence: a blue bike parked on a side walk.
                        Adjectives: ["blue"]
                        Selected adjective: blue
                        New adjective: red
                        New sentence: a red bike parked on a side walk.
                        Original sentence: The kitchen is clean and ready for us to see."""},
                {"role": "assistant", "content": """Adjectives: ["clean", "ready"]
                            Selected adjective: clean
                            New adjective: dirty
                            New sentence: The kitchen is dirty and ready for us to see.\n"""}],
                'REPLACE_REL':
                        [{"role": "user", "content": """Given an input sentence describing a scene, your task is to:
                            1. Find any action or spatial relationships between two objects in the sentence. If there are no such relationships, return an empty list.
                            2. Randomly pick one relationship.
                            3. Replace the selected relationship with a new relationship to make a new sentence.
                        The new sentence must meet the following three requirements:
                            1. The new sentence must be describing a scene that is as different as possible from the original scene.
                            2. The new sentence must be fluent and grammatically correct.
                            3. The new sentence must make logical sense.
                        Here are some examples:
                        Original sentence: The dining table near the kitchen has a bowl of fruit on it.
                        Relationships: ["near", "on"]
                        Selected relationship: near
                        New relationship: far from
                        New sentence: The dining table far from the kitchen has a bowl of fruit on it.
                        Original sentence: A couple of buckets in a white room."""},
                {"role": "assistant", "content": """Relationships: ['in']
                        Selected relationship: in
                        New relationship: outside
                        New sentence: A couple of buckets outside a white room.\n"""}],
                'SWAP_OBJ':
                        [{"role": "user", "content": """Given an input sentence describing a scene, your task is to first locate two swappable noun phrases in the sentence, and then swap them to make a new sentence.
                        The new sentence must meet the following three requirements:
                            1. The new sentence must be describing a different scene from the input sentence.
                            2. The new sentence must be fluent and grammatically correct.
                            3. The new sentence must make logical sense.
                        To complete the task, you should:
                            1. Answer the question of whether generating such a new sentence is possible using Yes or No.
                            2. Output the swappable noun phrases.
                            3. Swap them to make a new sentence.
                        Here are some examples:
                        Input: A cat resting on a laptop next to a person.
                        Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? 
                        Yes
                        Swappable noun phrases: laptop, person
                        Output: A cat resting on a person next to a laptop.
                        Input: A plate of donuts with a person in the background.
                        Is it possible to swap noun phrases in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense?"""},
                {"role": "assistant", "content": """Yes
                        Swappable noun phrases: a plate of donuts, a person
                        Output: A person with a plate of donuts in the background.\n"""}],
                'SWAP_ATT':
                        [{"role": "user", "content": """Given an input sentence describing a scene, your task is to first locate two swappable adjectives in the sentence describing different objects, and then swap them to make a new sentence.
                        The new sentence must meet the following three requirements:
                            1. The new sentence must be describing a different scene from the input sentence.
                            2. The new sentence must be fluent and grammatically correct.
                            3. The new sentence must make logical sense.
                        To complete the task, you should:
                            1. Answer the question of whether generating such a new sentence is possible using Yes or No.
                            2. Output the swappable adjectives.
                            3. Swap them to make a new sentence.
                        Here are some examples:
                        Input: A girl in a pink shirt holding a blue umbrella.
                        Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense? 
                        Yes
                        Swappable attributes: pink, blue
                        Output: A girl in a blue shirt holding a pink umbrella.
                        Input: A girl with a green shirt brushing her teeth with a blue toothbrush.
                        Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense?"""},
                {"role": "assistant", "content": """Yes
                        Swappable attributes: green, blue
                        Output: A girl with a blue shirt brushing her teeth with a green toothbrush.\n"""}],
                'ADD_OBJ':
                        [{"role": "user", "content": """Given an input sentence describing a scene, your task is:
                            1. Find the objects in the sentence.
                            2. Randomly pick one object.
                            3. Generate a new object that's not in the sentence.
                            4. Add the new object next to the selected object to make a new sentence.
                        The new sentence must meet the following three requirements:
                            1. The new sentence must describe a clearly new and different scene.
                            2. The new sentence must be fluent and grammatically correct.
                            3. The new sentence must make logical sense.
                        Here are some examples:
                        Original sentence: An elephant standing under the shade of a tree.
                        Objects: ["elephant", "shade of a tree"]
                        Selected object: elephant
                        New object: squirrel
                        New sentence: An elephant and a squirrel standing under the shade of a tree.
                        Original sentence: A bench at the beach next to the sea"""},
                {"role": "assistant", "content": """Objects: ['bench', 'beach', 'sea']
                        Selected object: bench
                        New object: umbrella
                        New sentence: An umbrella and a bench at the beach next to the sea.\n"""}],
                'ADD_ATT':
                        [{"role": "user", "content": """Given an input sentence describing a scene, your task is:
                            1. Find the objects in the sentence.
                            2. Randomly pick one object.
                            3. Generate a new plausible but uncommon attribute for this object that's not in the sentence.
                            4. Add the new attribute next to the selected object to make a new sentence.
                        The new sentence must meet the following three requirements:
                            1. The new sentence must describe a clearly new and different scene.
                            2. The new sentence must be fluent and grammatically correct.
                            3. The new sentence must make logical sense.
                        Here are some examples:
                        Original sentence: A large white airplane and a person on a lot.
                        Objects: ["airplane", "person"]
                        Selected object: airplane
                        New attribute: blue
                        New sentence: A large white and blue airplane and a person on a lot.
                        Original sentence: three people riding horses on a beach 
                        """},
                {"role": "assistant", "content": """Objects: ['three people', 'horses', 'beach']
                        Selected object: three people
                        New attribute: elderly
                        New sentence: Three elderly people riding horses on a beach.\n"""}]
                }

    if args.type[0] == 'all':
        subtypes = templates
    else:
        subtypes = dict((k, templates[k]) for k in (args.type))

    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

    for subtype in subtypes:
        data = source_data
        mess = []
        for ind in data.index:
            messages = list(subtypes[subtype])
            if (subtype == 'SWAP_OBJ') | (subtype == 'SWAP_ATT'):
                messages.append({"role": "user", "content": f" Input: {data['caption'][ind]} Is it possible to swap attributes in the input sentence to generate a new sentence that is different from the input sentence and makes logical sense?"})
            else:
                messages.append({"role": "user", "content": f"Original sentence: {data['caption'][ind]}"})
            prompt = pipe.tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
            mess.append(prompt)

        df = pd.DataFrame(list(mess), columns=['prompts'])
        dataset = Dataset.from_pandas(df)
        results = pipe(KeyDataset(dataset, "prompts"), max_new_tokens=350, do_sample=True, batch_size=200) # for only generated text return_full_text=False
        data['negative_caption'] =  [c[0]['generated_text'] for c in list(results)]

        for outputs in data.itertuples():
            if (subtype == 'SWAP_OBJ') | (subtype == 'SWAP_ATT'):
                if ((outputs[3].split("Correct Assistant:",2)[2][:4] == ' Yes')|(outputs[3].split("Correct Assistant:",2)[2][:3] == ('Yes') )) & (len(outputs[3].split("Output: ",3)) == 4):               
                    data['negative_caption'].iloc[outputs[0]] = outputs[3].split("Output: ",3)[3] 
                else:
                    data['negative_caption'].iloc[outputs[0]] = 'No'
            else:
                if len(outputs[3].split("New sentence: ",3)) == 4:
                    data['negative_caption'].iloc[outputs[0]] = outputs[3].split("New sentence: ",3)[3] 
                else:
                    data['negative_caption'].iloc[outputs[0]] = 'No'

        # Delete instances that do not follow the template   
        data = data[data['negative_caption'] != 'No']
        data = data.dropna()
        data['negative_caption'] = data['negative_caption'].apply(lambda x: x.split('\n')[0])
        data.to_csv(os.path.join(args.save_path, f'TROHN-TEXT_{subtype}.csv'), index = False, quotechar='"', quoting=csv.QUOTE_ALL)
    
    if args.type[0] == 'all':
        for subtype in subtypes:
            df = pd.read_csv(os.path.join(args.save_path, f'TROHN-TEXT_{subtype}.csv'))
            globals() [subtype] = df
        sub_df = [REPLACE_OBJ,REPLACE_ATT, REPLACE_REL ,SWAP_OBJ,SWAP_ATT,ADD_OBJ, ADD_ATT]
        data = pd.concat(sub_df).reset_index(drop = True)
        data.to_csv(os.path.join(args.save_path, f'TROHN-TEXT.csv'), index = False, quotechar='"', quoting=csv.QUOTE_ALL)

    