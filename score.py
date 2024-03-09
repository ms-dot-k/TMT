from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import argparse
import os, glob, re

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def main():
    parser = get_parser()
    args = parser.parse_args()

    dataDir = args.ref
    dataType ='val2014'
    annFile = os.path.join(dataDir, 'annotations/captions_val2014.json')

    preds = glob.glob(os.path.join(args.pred, '*.txt'), recursive=True)

    # create coco object and cocoRes object
    coco = COCO(annFile)

    res = []
    for p in preds:
        file_name = os.path.basename(p)[:-4]
        img_id = int(file_name.split('_')[-1])
        with open(p, 'r') as txt:
            try:
                text = txt.readlines()[0].strip()
            except:
                text = ' '
        res.append({"image_id": img_id, "caption": text})
    
    print('Prediction Num: ', len(res))
    with open(os.path.join(args.pred, 'pred.json'), "w") as outfile:
        json.dump(res, outfile)

    resFile = os.path.join(args.pred, 'pred.json')

    cocoRes = coco.loadRes(resFile)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    # print output evaluation scores
    with open(args.pred + '_score_coco.txt', 'w') as w:
        for metric, score in cocoEval.eval.items():
            w.write('%s: %.3f\n'%(metric, score))
            print('%s: %.3f'%(metric, score))

def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for scoring."
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="prediction"
    )
    parser.add_argument(
        "--ref", type=str, required=True, help="reference"
    )
    parser.add_argument(
        "--data", type=str, default='coco', help="reference"
    )
    return parser

def remove_special_characters(text):
    text = re.sub(chars_to_ignore_regex, '', text).lower()
    if text[-1] == ' ':
        text = text[:-1]
    return text

if __name__ == "__main__":
    main()