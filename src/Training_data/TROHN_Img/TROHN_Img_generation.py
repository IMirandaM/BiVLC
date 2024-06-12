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
n_images = 1
def get_inputs(prompts):
    num_images = n_images
    prompt = list(prompts)
    guidance_scale=5 # 0,2,5
    return {"prompt": prompt,  "num_images_per_prompt": num_images , "guidance_scale" : guidance_scale} 

# Train
df = pd.read_csv('./src/Training_data/TROHN_Img/data/TROHN-Img_generation.csv')
print(len(df))
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
            #distributed_state.wait_for_everyone()
        for i in range(0, (len(image)), n_images):
            index = prompt['index'][batch_index].item()
            image[i].resize((512,512)).save(f'./src/Training_data/TROHN_Img/imgs/{index}_{prompt["image_id"][batch_index][:-4]}_0.jpg')
            batch_index += 1
        torch.cuda.empty_cache()

         
df['negative_image'] = [f'{index}_{df["image_id"][index][:-4]}.jpg' for index in df.index]

df.to_csv('./src/Training_data/TROHN_Img/TROHN-Img.csv', columns = ["image_id","caption","negative_caption","negative_image"], index = False, quotechar='"', quoting=csv.QUOTE_ALL)