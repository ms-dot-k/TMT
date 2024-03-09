import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import re
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

class AVTDataset(Dataset):
    def __init__(
            self, 
            coco_path=None, 
            flickr_path=None, 
            spcoco_path=None, 
            spflickr_path=None, 
            spcoco_split_path=None, 
            karpathy_split_path=None, 
            mode='train', 
            num_im_unit=8192, 
            max_sp_len=1024,
            max_txt_len=1024,
            train_data='coco',
            test_data='coco',
            tokenizer="bert-large-uncased",
            image_tokenizer=None,
            test_asr=False,
            architecture='bert'
            ):
        assert mode in ['train', 'test', 'val']
        assert architecture in ['bert']
        self.mode = mode
        self.test_data = test_data
        self.train_data = train_data
        self.test_asr = test_asr
        self.architecture = architecture

        if train_data=='coco':
            self.f_paths = None
            self.im_paths, self.sp_paths, self.txt_paths = self.build_coco_file_list(
                [coco_path, flickr_path], 
                [spcoco_path, spflickr_path], 
                [spcoco_split_path, karpathy_split_path],
                mode, 
            )   #path_list, path_list, txt_list
        else:
            raise NotImplementedError
                
        self.max_sp_len = max_sp_len
        self.max_txt_len = max_txt_len

        self.num_sp_unit = 200
        
        self.unit_to_ind = None
        self.ind_to_unit = None
        self.codebook = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        ## Bert Tokenizer ##
        self.pad = self.tokenizer.convert_tokens_to_ids('[PAD]')  # 0 from BERT
        self.bos = self.tokenizer.convert_tokens_to_ids('[CLS]')  # 101 from BERT
        self.eos = self.tokenizer.convert_tokens_to_ids('[SEP]')  # 102 from BERT
        self.num_txt = len(self.tokenizer)

        self.num_sp_unit = self.num_sp_unit + 3
        self.num_im_unit = num_im_unit + 3

        self.image_tokenizer = image_tokenizer
        self.image_size = 224
        
        transform = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        if self.image_tokenizer == 'SEED':
            transform.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))
        self.transform = transforms.Compose(transform)

    def build_coco_file_list(self, image_path, speech_unit_path, split_path, mode):
        im_files, sp_files, txt_files = [], [], []
        co_im, fl_im = image_path
        co_sp, fl_sp = speech_unit_path
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

        if mode != 'test':
            data = json.load(open(os.path.join(kp_file, 'dataset_coco.json')))
            for d in data['images']:
                if d['split'] == mode:
                    im_path = os.path.join(co_im, d['filepath'], d['filename'][:-4] + '.jpg')
                    spcaptions = im2wav_mapping[f"{d['filepath']}/{d['filename'][:-4]}"][:5]
                    if mode == 'train':
                        for (cap, txt) in zip(*spcaptions):
                            sp_path = os.path.join(co_sp, cap + '.unit')
                            im_files.append(im_path)
                            sp_files.append(sp_path)
                            txt_files.append(txt.lower())
                    else:
                        cap, txt = list(zip(*spcaptions))[0]
                        sp_path = os.path.join(co_sp, cap + '.unit')
                        im_files.append(im_path)
                        sp_files.append(sp_path)
                        txt_files.append(txt.lower())

        elif self.test_data == 'coco':
            data = json.load(open(os.path.join(kp_file, 'dataset_coco.json')))
            for d in data['images']:
                if d['split'] == mode:
                    im_path = os.path.join(co_im, d['filepath'], d['filename'][:-4] + '.jpg')
                    spcaptions = im2wav_mapping[f"{d['filepath']}/{d['filename'][:-4]}"][:5]
                    if self.test_asr:
                        for (cap, txt) in zip(*spcaptions):
                            sp_path = os.path.join(co_sp, cap + '.unit')
                            im_files.append(im_path)
                            sp_files.append(sp_path)
                            txt_files.append(txt.lower())
                    else:
                        cap, txt = list(zip(*spcaptions))[0]
                        sp_path = os.path.join(co_sp, cap + '.unit')
                        im_files.append(im_path)
                        sp_files.append(sp_path)
                        txt_files.append(txt.lower())    # for validation purpose, only use the first caption

        num_coco = len(im_files)
        print(f"num COCO {num_coco}")

        if mode == 'train':
            data = json.load(open(os.path.join(kp_file, 'dataset_flickr8k.json')))
            for d in data['images']:
                if d['split'] == mode:
                    im_path = os.path.join(fl_im, d['filename'][:-4] + '.jpg')
                    captions = d['sentences']
                    for cap in range(5):
                        sp_path = os.path.join(fl_sp, d['filename'][:-4] + f'_{cap}.unit')
                        if os.path.exists(sp_path):
                            im_files.append(im_path)
                            sp_files.append(sp_path)
                            txt_files.append(captions[cap]['raw'].lower())
        
        elif mode == 'test' and self.test_data == 'flickr':
            data = json.load(open(os.path.join(kp_file, 'dataset_flickr8k.json')))
            for d in data['images']:
                if d['split'] == mode:
                    im_path = os.path.join(fl_im, d['filename'][:-4] + '.jpg')
                    captions = d['sentences']
                    if self.test_asr:
                        for cap in range(5):
                            sp_path = os.path.join(fl_sp, d['filename'][:-4] + f'_{cap}.unit')
                            if os.path.exists(sp_path):
                                im_files.append(im_path)
                                sp_files.append(sp_path)
                                txt_files.append(captions[cap]['raw'].lower())
                    else:
                        sp_path = os.path.join(fl_sp, d['filename'][:-4] + f'_0.unit')
                        im_files.append(im_path)
                        sp_files.append(sp_path)
                        txt_files.append(captions[0]['raw'].lower())
        
        print(f"num Flickr {len(im_files) - num_coco}")
        return im_files, sp_files, txt_files

    def __len__(self):
        return len(self.im_paths)


    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        sp_path = self.sp_paths[idx]
        txt_path = self.txt_paths[idx]
        skip = False

        ### Text
        if self.mode == 'train' and self.train_data != 'coco':
            with open(txt_path, 'r') as text:
                txt = text.readlines()[0].strip()
        else:
            txt = txt_path
        txt = self.text_normalization(txt)
        txt = self.tokenizer(text=txt, return_tensors="pt")
        txt = txt.input_ids.squeeze(0)  #Already attached bos and eos of tokenizer
            
        if len(txt) > self.max_txt_len:
            print(f'Skipping this sample due to long input length, txt:{len(txt)}')
            skip = True

        ### Speech
        try:
            sp_unit = torch.load(sp_path)
        except:
            sp_unit = [100] * 10
            skip = True
            print('Error while loading speech unit')

        sp_unit = self.process_units(sp_unit)  #remove repetition
        if len(sp_unit) > self.max_sp_len:
            print(f'Skipping this sample due to long input length, sp:{len(sp_unit)}')
            skip = True

        ### Add special token to speech unit
        sp_unit = self.add_special_room(sp_unit)
        sp_unit = self.append_eos(sp_unit)
        sp_unit = self.append_bos(sp_unit)
            
        ### Image
        try:
            image = Image.open(im_path).convert('RGB')
            im_input = self.transform(image)
        except:
            im_input = torch.zeros([3, self.image_size, self.image_size])
            skip = True
            print('Error while loading image')
        
        f_name = os.path.splitext(os.path.basename(im_path))[0] if not self.test_asr else os.path.splitext(os.path.basename(sp_path))[0]
        return im_input, sp_unit, txt, f_name, skip

    def process_units(self, units, char_map=None):
        if char_map is None:
            out = [int(u) for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
            return torch.tensor(out)
        else:
            out = [char_map[int(u)] for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
            return ''.join(out)

    def add_special_room(self, units):
        # self.pad # 0 in BERT tokenizer
        # self.bos # 101
        # self.eos # 102
        assert self.bos == 101 and self.eos == 102, 'bos is 101 and eos is 102 in bert tokenizer'
        if torch.is_tensor(units):
            units = units.numpy()
        # orig 0 will be 1 / orig 99 will be 100 / orig 100 will be 103 / orig 101 will be 104
        units[np.where(units >= 100)] += 2 #BOS & EOS room
        units += 1  #PAD room
        return torch.tensor(units)

    def del_special_room(self, units):
        assert self.bos == 101 and self.eos == 102, 'bos is 101 and eos is 102 in bert tokenizer'
        if torch.is_tensor(units):
            units = units.numpy()
        
        # minus values will be regarded as special tokens
        units[np.where(units == self.pad)] = 0
        units[np.where(units == self.bos)] = 0
        units[np.where(units == self.eos)] = -100
        # 1 will be 0 / 100 will be 99 / 103 will be 100 / 104 will be 101
        units -= 1 #Del PAD room
        units[np.where(units >= 102)] -= 2 #Del BOS & EOS room
        return units
            
    def text_normalization(self, text):
        text = re.sub(r"[\(\[].*?[\)\]]", "", text) # remove Coding conventions
        text = re.sub(r"[^\w\s']|_", " ", text) # remove punctuation except apostrophe
        text = re.sub(r"\s{2,}", " ", text).strip() # remove double space
        return text.lower()

    def append_bos(self, units):
        if len(units.size()) == 2:
            return torch.cat([torch.tensor([self.bos]).unsqueeze(0).repeat(units.size(0), 1), units], 1)
        else:
            return torch.cat([torch.tensor([self.bos]), units], 0)
    
    def append_eos(self, units):
        if len(units.size()) == 2:
            return torch.cat([units, torch.tensor([self.eos]).unsqueeze(0).repeat(units.size(0), 1)], 1)
        else:
            return torch.cat([units, torch.tensor([self.eos])], 0)

    def collate_fn(self, batch):
        sp_len, txt_len, f_names = [], [], []
        for data in batch:
            if not data[4]:
                sp_len.append(len(data[1]))
                txt_len.append(len(data[2]))
                f_names.append(data[3])

        max_txt_len = max(txt_len)
        max_sp_len = max(sp_len)

        im_inputs = []
        padded_sp_unit = []
        padded_txt = []

        for im_input, sp_unit, txt, _, skip in batch:
            if not skip:
                im_inputs.append(im_input)
                padded_sp_unit.append(torch.cat([sp_unit, torch.ones([max_sp_len - len(sp_unit)]) * self.pad], 0))
                padded_txt.append(torch.cat([txt, torch.ones([max_txt_len - len(txt)]) * self.pad], 0))

        im_inputs = torch.stack(im_inputs, 0)
        sp_len = torch.IntTensor(sp_len)
        txt_len = torch.IntTensor(txt_len)
        sp_unit = torch.stack(padded_sp_unit, 0).long()
        txt = torch.stack(padded_txt, 0).long()
        return torch.FloatTensor(im_inputs), torch.LongTensor(sp_unit), torch.LongTensor(txt), sp_len, txt_len, f_names
