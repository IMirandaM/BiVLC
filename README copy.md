# BiVLC: Extending Vision-Language Compositionality Evaluation with Text-to-Image Retrieval

<p align="center">
   <a href="https://imirandam.github.io/BiVLC_project_page"> Project Page </a> |
   <a href="https://huggingface.co/datasets/imirandam/BiVLC"> BiVLC Dataset </a> |
   <a href="https://huggingface.co/datasets/imirandam/TROHN-Text"> TROHN-Text Dataset </a> |
   <a href="https://huggingface.co/datasets/imirandam/TROHN-Img"> TROHN-Img Dataset </a> |
   <a href=""> Paper </a> |
   <a href="./src/CLIP_fine_tuning/ckpt"> Model Checkpoints </a> |
</p>

This is the official implementation for the paper BiVLC: Extending Vision-Language Compositionality Evaluation with Text-to-Image Retrieval

WORK IN PROGRESS!

## Examples

## Results

## Code Structure

```
├── source_data                               # Directory for source data
│   ├── COCO_2017                             # Folder to save COCO 2017
│   ├── SugarCrepe                            # Folder to save SugarCrepe
│   │   └── concat_SugarCrepe.py              # SugarCrepe 7 json files into 1 csv
├── src                                       # Data generation and training code
│   ├── BiVLC_Generation                      # BiVLC data generation
│   │   ├── imgs                              # Folder to save the generated imgs
│   │   └── BiVLC_img_generation.py           # Image generation
│   ├── Training_data                         # Train and val data generation
│   │   ├── TROHN-Text                        # TROHN-Text data generation
│   │   │   ├── data                          # Generated negative captions
│   │   │   └── TROHN_Text_generation.py      # Negative caption generation 
│   │   └── TROHN-Img                         # TROHN-Img data generation
│   │   │   ├── data                          # Scored negative captions
│   │   │   ├── adversarial_refine.py         # Adversarial refinement from SugarCrepe
│   │   │   ├── TROHN_Img_scoring.py          # Scoring TROHN-Text negative captions
│   │   │   ├── TROHN_Img_best_selection.py   # Selecting best negative captions
│   │   │   └── TROHN_Img_generation.py       # Image generation
│   ├── CLIP_fine_tuning                      # CLIP fine-tuning code and ckpts
│   │   ├── ckpt                              # CLIP Checkpoints
│   │   ├── CLIP_fine_tuning.py               # CLIP fine-tuning
│   │   ├── CLIP_utils.py                     # load model, load data, evaluations
│   │   └── scheduler.py                      # Cosine scheduler from OpenCLIP
│   ├── Detectors                             # Detectors training and evaluation
│   │   ├── classifier_ckpt                   # Classifier checkpoints
│   │   ├── Detector_img_classifier.py        # Training img detector
│   │   ├── Detector_text_classifier.py       # Training text detector
│   │   ├── Detector_evaluation_BiVLC.py      # BiVLC detector evaluation
│   │   ├── Detector_evaluation_SugarCrepe.py # SugarCrepe detector evaluation
│   │   └── Detector_utils.py                 # load model, load data
├── main_evaluation_BiVLC.py                  # Main evaluation 
├── evaluation_SugarCrepe.py                  # SugarCrepe evaluation
└── requirements.txt                          # List of libraries needed

```

## Instructions for replicating the results
### Create Python Environment

```
venv BiVLC                        # Create env
source BiVLC/bin/activate         # Load env

pip install -r requirements.txt   # Install dependencies
```

### Download source data

#### COCO 2017
Download the COCO 2017 data from the official website, https://cocodataset.org/#download.
1. Download the COCO 2017 Train/Val annotations and save captions_train2017.json in the BiVLC/source_data/COCO_2017 directory.
2. Download the COCO 2017 train and val images and save the unzipped folders in the data_source/COCO_2017 directory as train2017 and val2017.

Resulting in:
```
├── source_data                               
│   └── COCO_2017                              
│   │   ├── captions_train2017.json           
│   │   ├── train2017
│   │   └── val2017
```

#### SugarCrepe
Download SugarCrepe data in BiVLC/source_data/SugarCrepe from the official repository, https://github.com/RAIVNLab/sugar-crepe/tree/main/data. 

Then we concatenate the 7 json files into a single CSV.

```
python source_data/SugarCrepe/concat_SugarCrepe.py
```

### Data generation

#### BiVLC

To create BiVLC we relied on SugarCrepe negative captions and created 4 images for each caption with SD-XL model. Then, through two phases of crowdsourcing, we kept the best images (see dataset curation section).

```
accelerate launch --num_processes=4 src/BiVLC_Generation/BiVLC_img_generation.py
```
**Note:** We have used 4 GPUs for the execution, if you want to use a different number modify --num_processes= number of GPUs.

#### TROHN-Text

To create TROHN-Text we have used the COCO 2017 train captions, using OpenChat 3.5-0106 and the templates provided by Sugarcrepe. We have created a negative caption for the proposed subcategories in SugarCrepe for each of the COCO 2017 train captions.

```
python src/Training_data/TROHN_Text/TROHN_Text_generation.py
```

#### TROHN-Img

To create TROHN-Img we used the captions generated for TROHN-Text. As image generation requires a lot of computational power and time, we filtered the negative captions based on plausibility and linguistic acceptability scores to obtain the best ones.

```
python src/Training_data/TROHN_Img/TROHN_Img_scoring.py
python src/Training_data/TROHN_Img/TROHN_Img_best_selection.py
```

Once we have the best captions, we generate an image for each caption with the SD-XL model.

```
accelerate launch --num_processes=6 src/Training_data/TROHN_Img/TROHN_Img_generation.py
```
**Note:** We have used 6 GPUs for the execution, if you want to use a different number modify --num_processes= number of GPUs.

### CLIP fine-tuning

We have fine-tuned 3 CLIP models with different data, COCO 2017 train, TROHN-Text and TROHN-Img. 

```
python src/CLIP_fine_tuning/CLIP_fine_tuning.py \
--dataset 'COCO' --run_name CLIP_COCO
```
**Note:** In the above code, we fine-tuned CLIP with the COCO 2017 training dataset, change --dataset to **'TROHN-Text'** or **'TROHN-Img'** for the other two fine-tunings. Also, change --run_name to CLIP_TROHN-Text or CLIP_TROHN-Img to be able to identify the checkpoints correctly.

### Detectors

#### Training
We have trained natural vs synthetic image and text detectors.
To train the text detector.
```
python src/Detectors/Detector_text_classifier.py \
--model-checkpoint 'src/CLIP_fine_tuning/ckpt/best_CLIP_TROHN-Img_9.pt' \
--run_name 'CLIP_TROHN-Img'
```
To train the image detector.
```
python src/Detectors/Detector_img_classifier.py \
--model-checkpoint 'src/CLIP_fine_tuning/ckpt/best_CLIP_TROHN-Img_9.pt' \
--run_name 'CLIP_TROHN-Img'
```

**Note:** To train with the baseline model encoders simply do not add the --model-checkpoint argument.

```
python src/Detectors/Detector_text_classifier.py \
python src/Detectors/Detector_img_classifier.py \
```

#### Evaluate in BiVLC

We evaluated the detectors in BiVLC.
```
python src/Detectors/Detector_evaluation_BiVLC.py \
--image-checkpoint 'src/Detectors/classifier_ckpt/best_image_CLIP_TROHN-Img_0.pt' \
--text-checkpoint 'src/Detectors/classifier_ckpt/best_text_CLIP_TROHN-Img_6.pt' \
--run_name 'CLIP_TROHN-Img'
```

**Note:** In the previous example we evaluated the model based on the CLIP_TROHN-Img encoders, to evaluate the detector with the baseline encoders change the image-checkpoint and text-checkpoint by the checkpoints provided in [classifier ckpt](./src/Detectors/classifier_ckpt).

```
python src/Detectors/Detector_evaluation_BiVLC.py \
--image-checkpoint 'src/Detectors/classifier_ckpt/best_image_C_VIT-B-32_DPO_0.pt' \
--text-checkpoint 'src/Detectors/classifier_ckpt/best_text_C_VIT-B-32_7.pt' \
--run_name 'CLIP_baseline'
```

#### Evaluate in SugarCrepe

We evaluated text detectors in SugarCrepe. To evaluate detector CLIP_TROHN-Img.
```
python src/Detectors/Detector_evaluation_SugarCrepe.py \
--text-checkpoint 'src/Detectors/classifier_ckpt/best_text_CLIP_TROHN-Img_6.pt' \
--run_name 'CLIP_TROHN-Img'
```

As when evaluating in BiVLC, to evaluate the detector with the baseline model encoders just change the text-checkpoint and the run_name.

```
python src/Detectors/Detector_evaluation_SugarCrepe.py \
--text-checkpoint 'src/Detectors/classifier_ckpt/best_text_C_VIT-B-32_7.pt' \
--run_name 'CLIP_baseline'
```

### Evaluation
#### Main evaluation BiVLC

```
python main_evaluation_BiVLC.py \
--model-checkpoint 'src/CLIP_fine_tuning/ckpt/best_CLIP_TROHN-Img_9.pt' \
--run_name 'CLIP_TROHN-Img'
```

**Note:** In the above example we only evaluate CLIP_TROHN-Img, to evaluate any other model just change the --model-checkpoint argument and add a --run_name to identify it. In the [ckpt folder](./src/CLIP_fine_tuning/ckpt), we provide the checkpoints of our models, for [NegCLIP](https://github.com/mertyg/vision-language-models-are-bows) and [GNM](https://github.com/ugorsahin/Generative-Negative-Mining) models you should download their checkpoints directly from their official repositories. To evalute the baseline model simply do not add the --model-checkpoint argument.

#### Evaluate in SugarCrepe

To evaluate the different models in SugarCrepe change the model checkpoints and run names as in the BiVLC evaluation.

```
python evaluation_SugarCrepe.py \
--model-checkpoint 'src/CLIP_fine_tuning/ckpt/best_CLIP_TROHN-Img_9.pt' \
--run_name 'CLIP_TROHN-Img'
```

## License
See the [LICENSE](./LICENSE) file for details about the license under which code and data is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{,
  title={},
  author={},
  journal={},
  year={2024}
}
