# TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages

This repository contains the PyTorch implementation of the following paper:
> **TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages**<br>
> Minsu Kim*, Jee-weon Jung*, Hyeongseop Rha, Soumi Maiti, Siddhant Arora, Xuankai Chang, Shinji Watanabe, and Yong Man Ro (*equal contribution)<br>
> \[[Paper](https://arxiv.org/abs/2402.16021)\]

<div align="center"><img width="90%" src="imgs/img.png?raw=true" /></div>

## Requirements
- python 3.9
- pytorch 2.12
- ffmpeg
- tensorboard
- opencv-python
- pillow
- librosa
- editdistance
- transformers 4.35.2
- xformers 0.0.23
- timm
- einops 0.7.0
- fairseq
- sacrebleu
- diffusers 0.20.2
- bitarray
- accelerate

### Datasets Download
The training code example only uses COCO and Flickr8k.
If you want to train the model with large dataset (CC12M and CC3M), please download datasets by referring to https://github.com/rom1504/img2dataset. And write your own dataset loader.

SpokenCOCO dataset
- https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi

Flickr8k Audio dataset
- https://groups.csail.mit.edu/sls/downloads/flickraudio/

COCO2014 dataset
- https://cocodataset.org/

Flickr8k dataset
- https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names

Karpathy split
- https://cs.stanford.edu/people/karpathy/deepimagesent/

### Directory structure
```
COCO_2014
├── annotations
|   └── *.json
├── train2014
|   └── *.jpg
├── val2014
└── test2014

SpokenCOCO
├── *.json
└── Hubert_units
    ├── train
    |   └── *
    |       └── *.unit
    └── val

Flickr8k
├── Images
|   └── *.jpg
└── captions.txt

Flickr8k_audio
└── Hubert_units
    └── *.unit
```

## 1. Extracting Speech Unit
We directly utilized the pre-trained K-means cluster model of [link](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit).
Please refer to the repository to extract speech unit (HuBERT Base + KM200).
We use different output format with the repository (.unit), we save each speech unit by using
```
feat = FR.get_feature(file)
pred = kmeans_model.predict(feat)
pred = np.asarray(pred, dtype=np.int64)
torch.save(pred, out_path + '.unit')
```
Please put the extracted units as the above directory structure.

## 2. Unit-based Vocoder
Please download pre-trained unit-based vocoder [CKPT](https://drive.google.com/file/d/15MVGWCfvt_ZjQwBR4sizQKFMb5NPQ3Qy/view?usp=drive_link) and [CFG](https://drive.google.com/file/d/15MOiXjrUJJHMdIF6bxCWIxvWwUw6z7Kj/view?usp=sharing).
Put `g_00950000` and `config.json` in `./Vocoder/` directory.

## 3. Image Unit Extractor and Decoder
We employ [SEED-2 tokenizer](https://github.com/AILab-CVC/SEED) and decoder.
Please download [seed_quantizer.pt](https://huggingface.co/AILab-CVC/seed-tokenizer-2) and put the checkpoint in `./pretrained/`

After cloning this repository, call `batch_im_gen_enable.py` to enable the batch decoding of images.

```shell
git submodule init
git submodule update
python batch_im_gen_enable.py
```

## 4. Demo
Test Demo Jupyter Notebook for each task can be found at `demo.ipynb`.
Pre-trained model is available, please find it below.

## Testing the pre-trained Model
To test the model, modify some argument in `test.sh`. <br>
Please refer below for the argument information.
After properly setting the argument, run following command:
```shell
# test example
sh test.sh
```

The command will run the tests for 6 tasks, image captioning, iamge-to-speech captioning, text-to-image synthesis, speech-to-image synthesis, speech-to-text, and text-to-speech.

It takes almost 1 day, mainly the image generation takes time.
You can test on subset of tasks which is described below.

Descriptions of important argument:
```shell
################### SET ##################
SAVENAME=TMT_test
CKPT=./data/checkpoints/xx.ckpt
DEVICE=0

# Data
COCO=path_to/COCO_2014
FLICKR=path_to/Flickr8k/Images
SPCOCO_U=path_to/SpokenCOCO/Hubert_units
SPFLICKR=path_to/flickr_audio/Hubert_units
SPCOCO=path_to/SpokenCOCO
KARPATHY=path_to/Karpathy_split
##########################################
```

- `SAVENAME`: The output directory
- `CKPT`: Model checkpoint
- `DEVICE`: GPU ID
- `COCO`: data dir path to COCO2014 ('dir_to/COCO_2014')
- `FLICKR`: data dir path to Flickr8k ('dir_to/Flickr8k/Images')
- `SPCOCO_U`: data dir path to Speech unit of SpokenCOCO ('dir_to/SpokenCOCO/Hubert_units')
- `SPFLICKR`: data dir path to Speech unit of Flickr8k ('dir_to/Flickr8k_audio/Hubert_units')
- `SPCOCO`: data dir path to SpokenCOCO ('dir_to/SpokenCOCO', assuming the directory contains json files)
- `KARPATHY`: data dir path to Karpathy split ('dir_to/Karpathy_split')

### Test on fewer tasks
You can also test the model on some selected tasks by changing the `test.sh`.
For example, we can only test for captioning tasks, image captioning and image-to-speech captioning, by running `test_cap.sh`.
```shell
# test example
sh test_cap.sh
```

## Pre-trained model checkpoints
The pre-trained TMT model is available on [here](https://drive.google.com/file/d/15D3_gU2ajhF6ZfWPI8mFoqSM5_e6hgfO/view?usp=sharing). <br>
The model is trained on CC3M, CC12M, ImageNet-1k, CommonVoice, COCO, and Flickr8k. <br>
Please put the checkpoint in `data/checkpoints/`.

## Training the Model
To train the model, modify some argument in `train.sh`. <br>
Please refer below for the argument information.
After properly setting the argument, run following command:

```shell
# training example (Distributed training)
sh train.sh
```

Descriptions of training parameters are as follows:
- `GPUS`: GPU IDs
- `--project`: If set with some string, Wandb logging is available
- `--coco_path`: data dir path to COCO2014 ('dir_to/COCO_2014')
- `--flickr_path`: data dir path to Flickr8k ('dir_to/Flickr8k/Images')
- `--spcoco_path`: data dir path to Speech unit of SpokenCOCO ('dir_to/SpokenCOCO/Hubert_units')
- `--spflickr_path`: data dir path to Speech unit of Flickr8k ('dir_to/Flickr8k_audio/Hubert_units')
- `--spcoco_split_path`: data dir path to SpokenCOCO ('dir_to/SpokenCOCO', assuming the directory contains json files)
- `--karpathy_split_path`: data dir path to Karpathy split ('dir_to/Karpathy_split')
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--temp_dir`: temp directory where the evaluation files will be saved
- `--batch_size`: batch size 
- `--eval_step`: steps to perform evaluation
- `--lr`: learning rate
- `--update_frequency`: gradient accumulation steps
- `--generation_step`: steps to generate image during training (`--generate_im` should be set)
- `--generate_im`: If it is set, image will be generated during training (super slow and consume large memory)
- `--warmup`: If it is set, warmup lr scheduling is performed
- `--tot_iters`: The total iteration for training.
- `--fp16`: Wheter perform fp16 training

- `--im_txt`: Include Image-to-Text Translation task
- `--im_sp`: Include Image-to-Speech Translation task
- `--txt_sp`: Include Text-to-Speech Translation task
- `--txt_im`: Include Text-to-Image Translation task
- `--sp_txt`: Include Speech-to-Text Translation task
- `--sp_im`: Include Speech-to-Image Translation task
- `--num_task`: number of tasks performing

- Refer to `train.py` for the other training parameters

The evaluation during training is performed for a subset of the validation dataset due to the heavy inference time. <br>
In order to evaluate the entire performance of the trained model run the test code (refer to "Testing the Model" section).


## Citation
If you find this work useful in your research, please cite the paper:
```
@article{kim2024tmt,
  title={TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages},
  author={Kim, Minsu and Jung, Jee-weon and Rha, Hyeongseop and Maiti, Soumi and Arora, Siddhant and Chang, Xuankai and Watanabe, Shinji and Ro, Yong Man},
  journal={arXiv preprint arXiv:2402.16021},
  year={2024}
}
```
