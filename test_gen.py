import argparse
import os, re
from tqdm import tqdm
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch.nn.functional as F

def build_coco_file_list(image_path, split_path, test_data):
    im_files, txt_files = [], []
    co_im, fl_im = image_path
    co_file, kp_file = split_path

    im2wav_mapping = {}
    data = json.load(open(os.path.join(co_file, f'SpokenCOCO_train.json'), 'r'))
    for d in data['data']:
        im_path = d['image'][:-4]    #train2014/~~.jpg or val2014/~~.jpg
        spcaptions = d['captions']
        caps = []
        txts = []
        for cap in spcaptions:
            txts.append(cap['text'])
            cap = cap['wav'].replace('wavs/', '')[:-4]
            caps.append(cap)
        im2wav_mapping[im_path] = (caps, txts)
    data = json.load(open(os.path.join(co_file, f'SpokenCOCO_val.json'), 'r'))
    for d in data['data']:
        im_path = d['image'][:-4]    #train2014/~~.jpg or val2014/~~.jpg
        spcaptions = d['captions']
        caps = []
        txts = []
        for cap in spcaptions:
            txts.append(cap['text'])
            cap = cap['wav'].replace('wavs/', '')[:-4]
            caps.append(cap)
        im2wav_mapping[im_path] = (caps, txts)

    if test_data == 'coco':
        data = json.load(open(os.path.join(kp_file, 'dataset_coco.json')))
        for d in data['images']:
            if d['split'] == 'test':
                im_path = os.path.join(co_im, d['filepath'], d['filename'][:-4] + '.jpg')
                spcaptions = im2wav_mapping[f"{d['filepath']}/{d['filename'][:-4]}"][:5]
                txt_5 = []
                im_files.append(im_path)
                for (cap, txt) in zip(*spcaptions):
                    txt_5.append(text_normalization(txt.lower()))
                txt_files.append(txt_5)

    num_coco = len(im_files)
    print(f"num COCO {num_coco}")
    
    if test_data == 'flickr':
        data = json.load(open(os.path.join(kp_file, 'dataset_flickr8k.json')))
        for d in data['images']:
            if d['split'] == 'test':
                im_path = os.path.join(fl_im, d['filename'][:-4] + '.jpg')
                captions = d['sentences']
                im_files.append(im_path)
                txt_5 = []
                for cap in range(5):
                    txt_5.append(text_normalization(captions[cap]['raw'].lower()))
                txt_files.append(txt_5)

    elif test_data =='flickr30k':
        data = json.load(open(os.path.join(kp_file, 'dataset_flickr30k.json')))
        for d in data['images']:
            if d['split'] == 'test':
                im_path = os.path.join(fl_im, d['filename'][:-4] + '.jpg')
                captions = d['sentences']
                im_files.append(im_path)
                txt_5 = []
                for cap in range(5):
                    txt_5.append(text_normalization(captions[cap]['raw'].lower()))
                txt_files.append(txt_5)

    print(f"num Flickr {len(im_files) - num_coco}")
    return im_files, txt_files

def main():
    parser = get_parser()
    args = parser.parse_args()

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    im_lists, txt_lists = build_coco_file_list(
        image_path=[args.coco_path, args.flickr_path],
        split_path=[args.spcoco_split_path, args.karpathy_split_path],
        test_data=args.data
    )

    cos_scores = []

    for im_path in tqdm(im_lists):
        f_name = os.path.basename(im_path)
        f_name = os.path.join(args.pred, f_name)
        gt_image = Image.open(im_path)
        pred_image = Image.open(f_name)
        gt_inputs = processor(images=gt_image, return_tensors="pt")
        pred_inputs = processor(images=pred_image, return_tensors="pt")

        gt_inputs = {k: v.cuda() for k, v in gt_inputs.items()}
        gt_feat = model.get_image_features(**gt_inputs)

        pred_inputs = {k: v.cuda() for k, v in pred_inputs.items()}
        pred_feat = model.get_image_features(**pred_inputs)

        cos_sim = F.cosine_similarity(gt_feat, pred_feat, dim=1)
        cos_sim = cos_sim.squeeze()

        cos_scores.append(cos_sim.cpu().item())

    print("CLIP_cosine_score : ", np.mean(cos_scores))
    with open(args.pred + '_CLIP_gt_score.txt', 'w') as w:
        w.write(f'Cosine: {str(np.mean(cos_scores))}')

def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for scoring."
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="prediction"
    )
    parser.add_argument(
        "--coco_path", type=str, default="path_to/COCO_2014", help="reference"
    )
    parser.add_argument(
        "--flickr_path", type=str, default="path_to/Flickr8k/Images", help="reference"
    )
    parser.add_argument(
        "--spcoco_split_path", type=str, default="path_to/SpokenCOCO", help="reference"
    )
    parser.add_argument(
        "--karpathy_split_path", type=str, default="path_to/Flickr8k/Karpathy_split", help="reference"
    )
    parser.add_argument(
        "--data", type=str, default='coco', help="reference"
    )
    return parser

def text_normalization(text):
    text = re.sub(r"[\(\[].*?[\)\]]", "", text) # remove Coding conventions
    text = re.sub(r"[^\w\s']|_", " ", text) # remove punctuation except apostrophe
    text = re.sub(r"\s{2,}", " ", text).strip() # remove double space
    return text.lower()

if __name__ == "__main__":
    main()