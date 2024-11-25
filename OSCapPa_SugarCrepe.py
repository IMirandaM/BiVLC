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
    #print(item['filename'])
    img = Image.open(os.path.join(img_path, item['filename']))
    img = img.resize((256, 256))
    img = img.convert("RGB")
    pixel_values = image_to_logits(img)
    pixel_values = pixel_values[np.newaxis, ...]
    # text
    pos_inputs = process_text(item["caption"])
    #print(f'Positive: {item["caption"]}')
    neg_inputs = process_text(item["negative_caption"])
    #print(f'Negative: {item["negative_caption"]}')
    return {
        "pixel_values": pixel_values,
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

    # load data
    print('Loading data...')
    data = pd.read_csv('/gaueko0/users/imiranda014/Esperimentuak/SugarCrepe/t2v_metrics/test_SugarCrepe.csv')
    print('Data loaded!')
    img_path = '/gaueko0/users/imiranda014/GSCRATCH/Data/COCO/val2017'

    # download model
    model_name = "boris/cappa-large-patch16-256-jax"
    local_dir = "/gaueko0/users/imiranda014/Esperimentuak/SugarCrepe/osCapPa/model/cappa-large-patch16-256-jax"
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

    metric = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        inputs = load_item(row)
        C0_I0 = get_scores(inputs["pixel_values"], inputs["pos_inputs"], params)
        C1_I0 = get_scores(inputs["pixel_values"], inputs["neg_inputs"], params)

        score = C0_I0 > C1_I0
        metric.append(score)

    data['score'] = [tensor.item() for tensor in metric]

    data_replace = data[data['type'] == 'replace']
    data_swap = data[data['type'] == 'swap']
    data_add = data[data['type'] == 'add']

    print(f"OS CapPa checkpoint OVERALL ({len(data)}) scores: {100*(sum(data['score'])/len(data)):.4}")
    print(f"OS CapPa checkpoint REPLACE ({len(data_replace)}) scores: {100*(sum(data_replace['score'])/len(data_replace)):.4}")
    print(f"OS CapPa checkpoint SWAP ({len(data_swap)}) scores: {100*(sum(data_swap['score'])/len(data_swap)):.4}")
    print(f"OS CapPa checkpoint ADD ({len(data_add)}) scores: {100*(sum(data_add['score'])/len(data_add)):.4}")