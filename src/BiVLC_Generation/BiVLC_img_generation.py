import pandas as pd
from tqdm import tqdm
import torch
import torchvision.transforms as T
import csv
from datasets import Dataset
from torch.utils.data import DataLoader
from diffusers import  StableDiffusionXLPipeline, UNet2DConditionModel
from accelerate import PartialState

print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

distributed_state = PartialState()

# SD-XL DPO
# Base DPO
# load pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
base = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")

# load finetuned model
unet_id = "mhdang/dpo-sdxl-text2image-v1"
unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
base.unet = unet

base.to(distributed_state.device)

generator = torch.Generator("cuda").manual_seed(42)
n_images = 4
def get_inputs(prompts):
    num_images = n_images
    prompt = list(prompts)
    guidance_scale=5
    return {"prompt": prompt,  "num_images_per_prompt": num_images , "guidance_scale" : guidance_scale} 

# Read SugarCrepe
df = pd.read_csv('./source_data/SugarCrepe/test_SugarCrepe.csv', usecols=["filename","caption","negative_caption","type","subtype"])
print(len(df))
df["prompt"] = (df["negative_caption"].str.replace(".", "")) + ", high resolution, professional, 4k, highly detailed" 
df['index'] = df.index
print(df.head())

dataset = Dataset.from_pandas(df)
print(dataset)
data = DataLoader(dataset, batch_size=32)

with torch.no_grad():
    pbar = tqdm(data, desc='Batch',)
    for batch in pbar:
        with distributed_state.split_between_processes(batch) as prompt:
            print(prompt)
            batch_index = 0
            image = base(**get_inputs(prompt["prompt"])).images
        for i in range(0, (len(image)), n_images):
            index = prompt['index'][batch_index].item()
            image[i].resize((512,512)).save(f'./src/BiVLC_Generation/imgs/{index}_{prompt["filename"][batch_index][:-4]}_0.jpg') 
            image[i+1].resize((512,512)).save(f'./src/BiVLC_Generation/imgs/{index}_{prompt["filename"][batch_index][:-4]}_1.jpg')
            image[i+2].resize((512,512)).save(f'./src/BiVLC_Generation/imgs/{index}_{prompt["filename"][batch_index][:-4]}_2.jpg')
            image[i+3].resize((512,512)).save(f'./src/BiVLC_Generation/imgs/{index}_{prompt["filename"][batch_index][:-4]}_3.jpg')
            batch_index += 1
        torch.cuda.empty_cache()
         
df['hn_img_0'] = [f'{index}_{df["filename"][index][:-4]}' for index in df.index] 
df['hn_img_1'] = df['hn_img_0'].astype(str) + '_1.jpg'
df['hn_img_2'] = df['hn_img_0'].astype(str) + '_2.jpg'
df['hn_img_3'] = df['hn_img_0'].astype(str) + '_3.jpg'
df['hn_img_0'] = df['hn_img_0'].astype(str) + '_0.jpg'
df = df.drop(['prompt'], axis=1)
df['caption'] = df['caption'].apply(lambda x: (x.strip().capitalize()) +'.' if x[-1] != '.' else x.strip().capitalize())
df['negative_caption'] = df['negative_caption'].apply(lambda x: (x.strip().capitalize()) +'.' if x[-1] != '.' else x.strip().capitalize())

#Test
df.to_csv('./src/BiVLC_Generation/BiVLC_negative_imgs.csv', index = False, quotechar='"', quoting=csv.QUOTE_ALL)
