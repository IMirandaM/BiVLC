import os
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

import jax
print(jax.devices())
import jax.numpy as jnp
import numpy as np
import optax
import orbax
#import wandb
from flax.training import orbax_utils
from flax.traverse_util import flatten_dict
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from clip_jax import CLIPModel
from clip_jax.data import image_to_logits, shift_tokens_left
#from clip_jax.tokenizer import AutoTokenizer
from clip_jax.utils import load_config

from functools import partial
from io import BytesIO
import subprocess

def download_huggingface_model(model_name, local_dir):
    # check if model is already download
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Model '{model_name}' is already downloaded in '{local_dir}'.")
        return

    # download
    command = f'huggingface-cli download {model_name} --local-dir {local_dir}'
    
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e.stderr.decode('utf-8')}")


def process_text(c):
    captions = " ".join(
                c.lower()
                .replace(",", ", ")
                .replace(".", ". ")
                .replace("-", " ")
                .replace(";", ", ")
                .replace(":", ", ")
                .replace('"', ' " ')
                .replace("/", ", ")
                .replace(".", ", ")
                .replace(")", ", ")
                .replace(" (", ", ")
                .strip(", ?\n")
                .split()
            ).replace(" ,", ",")
    txt_inputs = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=config["text_config"]["max_length"],
        return_tensors="np",
    )
    labels = shift_tokens_left(txt_inputs["input_ids"], pad_token_id=tokenizer.pad_token_id)
    labels_mask = shift_tokens_left(txt_inputs["attention_mask"], pad_token_id=0)
    return {
        "input_ids": txt_inputs["input_ids"],
        "attention_mask": txt_inputs["attention_mask"],
        "labels": labels,
        "labels_mask": labels_mask,
    }

def load_item(item):
    # pos_image
    #print(item['image'][0])
    img = item['image'][0]
    img = img.resize((256, 256))
    img = img.convert("RGB")
    pixel_values = image_to_logits(img)
    pixel_values = pixel_values[np.newaxis, ...]
    # neg_image
    #print(item['negative_image'][0])
    neg_img = item['negative_image'][0]
    neg_img = neg_img.resize((256, 256))
    neg_img = neg_img.convert("RGB")
    neg_pixel_values = image_to_logits(neg_img)
    neg_pixel_values = neg_pixel_values[np.newaxis, ...]
    # text
    pos_inputs = process_text(item["caption"][0])
    #print(f'Positive: {item["caption"][0]}')
    neg_inputs = process_text(item["negative_caption"][0])
    #print(f'Negative: {item["negative_caption"][0]}')
    return {
        "pixel_values": pixel_values,
        "neg_pixel_values": neg_pixel_values,
        "pos_inputs": pos_inputs,
        "neg_inputs": neg_inputs,
    }

@jax.jit
def get_scores(pixel_values, inputs, params):
    assert pixel_values.shape[0] == 1, "only support 1 image at a time"
    encoder_outputs = model.apply({"params": params}, pixel_values=pixel_values, method=model.get_image_features)[
        "vision_model_output"
    ]["last_hidden_state"]
    logits = model.apply(
        {"params": params},
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        encoder_hidden_states=encoder_outputs,
        decode=False,
        method=model.get_text_features,
    )["text_model_output"]["last_hidden_state"]
    score = -optax.softmax_cross_entropy_with_integer_labels(logits, inputs["labels"]) * inputs["labels_mask"]
    score = score.sum(axis=-1)
    return score[0]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    args = parser.parse_args()
    
    # download data
    print('Loading data...')
    data = load_dataset("imirandam/BiVLC", split = "test")
    print('Data loaded!')
    print(data)

    # download model
    model_name = "boris/cappa-large-patch16-256-jax"
    local_dir = args.local_dir
    download_huggingface_model(model_name, local_dir)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load model
    print(f'Loading model...')
    config_name = f"{local_dir}/config.json"
    config = load_config(config_name)

    model = CLIPModel(**config)
    rng = jax.random.PRNGKey(0)
    logical_shape = jax.eval_shape(lambda rng: model.init_weights(rng), rng)["params"]
    params = jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), logical_shape)
    # get model checkpoint
    model_path = str(Path(local_dir).resolve())
    model_path

    # restore checkpoint
    ckpt = {"params": params}
    restore_args = orbax_utils.restore_args_from_target(ckpt)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_options = orbax.checkpoint.CheckpointManagerOptions()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(model_path, orbax_checkpointer, orbax_options)
    step = checkpoint_manager.latest_step()
    ckpt = checkpoint_manager.restore(step, ckpt, restore_kwargs={"restore_args": restore_args, "transforms": {}})
    params = ckpt["params"]

    # ensure params have been set
    for k, v in flatten_dict(params).items():
        if jnp.sum(jnp.abs(v.value)) == 0:
            print(f"Warning: {k} has not been set")
    print('Model loaded!')
    
    iter=data.iter(batch_size=1)

    I2T = []
    T2I = []
    group_score = []

    Ipos_2T = []
    Ineg_2T = []
    Tpos_2I = []
    Tneg_2I = []

    for item in tqdm(iter):
        inputs = load_item(item)
        C0_I0 = get_scores(inputs["pixel_values"], inputs["pos_inputs"], params)
        C0_I1 = get_scores(inputs["neg_pixel_values"], inputs["pos_inputs"], params)
        C1_I0 = get_scores(inputs["pixel_values"], inputs["neg_inputs"], params)
        C1_I1 = get_scores(inputs["neg_pixel_values"], inputs["neg_inputs"], params)

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

    print(f'OS CapPa checkpoint OVERALL scores:   I2T: {100*(sum(I2T)/len(data)):.4}, T2I: {100*(sum(T2I)/len(data)):.4}, Group score: {100*(sum(group_score)/len(data)):.4}, Ipos_2T: {100*(sum(Ipos_2T)/len(data)):.4}, Ineg_2T: {100*(sum(Ineg_2T)/len(data)):.4}, Tpos_2I: {100*(sum(Tpos_2I)/len(data)):.4}, Tneg_2I: {100*(sum(Tneg_2I)/len(data)):.4}')

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

    print(f"OS CapPa checkpoint REPLACE ({len(df_replace)}) scores:   I2T: {100*(sum(df_replace['I2T'])/len(df_replace)):.4}, T2I: {100*(sum(df_replace['T2I'])/len(df_replace)):.4}, Group score: {100*(sum(df_replace['group_score'])/len(df_replace)):.4}, Ipos_2T: {100*(sum(df_replace['Ipos_2T'])/len(df_replace)):.4}, Ineg_2T: {100*(sum(df_replace['Ineg_2T'])/len(df_replace)):.4}, Tpos_2I: {100*(sum(df_replace['Tpos_2I'])/len(df_replace)):.4}, Tneg_2I: {100*(sum(df_replace['Tneg_2I'])/len(df_replace)):.4}")
    print(f"OS CapPa checkpoint SWAP ({len(df_swap)}) scores:   I2T: {100*(sum(df_swap['I2T'])/len(df_swap)):.4}, T2I: {100*(sum(df_swap['T2I'])/len(df_swap)):.4}, Group score: {100*(sum(df_swap['group_score'])/len(df_swap)):.4}, Ipos_2T: {100*(sum(df_swap['Ipos_2T'])/len(df_swap)):.4}, Ineg_2T: {100*(sum(df_swap['Ineg_2T'])/len(df_swap)):.4}, Tpos_2I: {100*(sum(df_swap['Tpos_2I'])/len(df_swap)):.4}, Tneg_2I: {100*(sum(df_swap['Tneg_2I'])/len(df_swap)):.4}")
    print(f"OS CapPa checkpoint ADD ({len(df_add)}) scores:   I2T: {100*(sum(df_add['I2T'])/len(df_add)):.4}, T2I: {100*(sum(df_add['T2I'])/len(df_add)):.4}, Group score: {100*(sum(df_add['group_score'])/len(df_add)):.4}, Ipos_2T: {100*(sum(df_add['Ipos_2T'])/len(df_add)):.4}, Ineg_2T: {100*(sum(df_add['Ineg_2T'])/len(df_add)):.4}, Tpos_2I: {100*(sum(df_add['Tpos_2I'])/len(df_add)):.4}, Tneg_2I: {100*(sum(df_add['Tneg_2I'])/len(df_add)):.4}")
