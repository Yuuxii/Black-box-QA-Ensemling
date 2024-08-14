# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
import collections
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset
import h5py
from copy import deepcopy

from src.param import args
from utils import load_obj_tsv
from pretrain.tag_data_utilis import create_tags
from lxrt.tokenization import BertTokenizer
from lxrt.h5_data import ImageFeatureDataset
from torch.utils.data.dataloader import DataLoader

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.


class VIZTorchDataset(Dataset):
    def __init__(self, train,  args):
        super().__init__()
        
        self.train = train

        if train == 'train':
            self.preds = json.load(open('data/vizwiz/abvlmo_all_answers_viz_val0.json', 'r'))
            self.gt = json.load(open('data/vizwiz/vizwiz_val0.json', 'rb'))
        elif train == 'val':
            self.preds = json.load(open('data/vizwiz/abvlmo_all_answers_viz_val1.json', 'r')) 
            self.gt = json.load(open('data/vizwiz/vizwiz_val1.json', 'rb'))
        elif train == 'test':
            self.preds = json.load(open('data/vizwiz/abvlmo_all_answers_viz_test.json', 'r'))
            self.gt = json.load(open('data/vizwiz/test.json', 'rb'))

        print('original len:', len(self.preds))
        if train == 'train' or train == 'val':
            used_data_num = round(len(self.preds)*args.use_amount/100)
            self.preds, self.gt = self.preds[:used_data_num], self.gt[:used_data_num]
        print('used len:', len(self.preds) )

        if args.get("add_tags", False):
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )
            from lxrt.symbolic_vocabulary import SymbolicVocab
            self.symbolic_vocab = SymbolicVocab(args.objects_vocab, args.attributes_vocab)

    def __len__(self):
        return len(self.preds)

    def __getitem__(self, item: int):
        
        datum = self.gt[item]
        
        albef = self.preds[item]['albef']
        blip = self.preds[item]['blip']
        vbert = self.preds[item]['vlmo']
        # print('question id:', datum['question_id'], albef['question_id'], blip['question_id'], vbert['question_id'])
        # vqa_gt = {datum['question_id']: datum}
        if args.dataname == 'vqa':
            img_id = datum['image'].split('/')[-1].split('.')[0]
        elif args.dataname == 'gqa':
            img_id = datum['image']
        ques = datum['question']
    

        if self.train == 'train' or self.train == 'val':
            img_dir = 'val'
        else:
            img_dir = 'test'

        # redefine np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        img_feature = np.load(os.path.join('visual_data/vizwiz/vizwiz_%s_feats'%(img_dir),  datum['image']+ '.npz'))
        
        np.load = np_load_old

        obj_num = img_feature['num_bbox'].item()
        feats = img_feature['x'].tolist()
        boxes = img_feature['bbox']
        img_h = img_feature['image_h'].item() 
        img_w = img_feature['image_w'].item()
        obj_labels = img_feature['info'].item()['objects_id'].tolist()
        attr_labels = img_feature['info'].item()['attrs_id'].tolist()      
  
        # Normalize the boxes (to 0 ~ 1)
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)


        if args.get("add_tags", False):
            tags = create_tags(obj_labels=obj_labels, attr_labels=attr_labels, obj_confs=None, attr_confs=None, tokenizer=self.tokenizer, symbolic_vocab = self.symbolic_vocab, visual_tags_box = boxes, use_bert_input=True)
        else:
            tags = None
        
        # Provide label (target)
        if 'answer' in datum:
            answer = datum['answer']
            # print('question id:', datum['question_id'], albef['question_id'], blip['question_id'], vbert['question_id'])
            # assert datum['question_id'] == albef['question_id'] == blip['question_id'] == vbert['question_id']
               
            if albef in answer:
                a_score = answer[albef]   
            else:
                a_score = 0.

            if blip in answer:
                b_score = answer[blip] 
            else:
                b_score = 0.
            
            if vbert in answer:
                v_score = answer[vbert] 
            else:
                v_score = 0.
            
            label = [a_score, b_score, v_score]
            

            return datum['question_id'], feats, boxes, tags, ques, label, [albef, blip, vbert]
        else:
            return datum['question_id'], feats, boxes, tags, ques, None, [albef, blip, vbert]


