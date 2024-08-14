# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
import collections
from typing_extensions import Self
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
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

class VIZDataset:
    def __init__(self, dataname: str, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        self.dataname = dataname
        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/%s/%s.json" % (dataname, split))))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/%s/trainval_ans2label.json" % dataname))
        self.label2ans = json.load(open("data/%s/trainval_label2ans.json" % dataname))
    
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

class VIZTorchDataset(Dataset):
    def __init__(self, dataset: VIZDataset, args):
        super().__init__()

        self.raw_dataset = dataset
        self.data = dataset.data
        if self.raw_dataset.splits[0] == 'vizwiz_val0' or self.raw_dataset.splits[0] == 'vizwiz_val1':
            used_data_num = round(len(self.data)*args.use_amount/100)
            self.data = self.data[: used_data_num] 
        print('used data len:', len(self.data))
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        if args.get("add_tags", False):
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )
            from lxrt.symbolic_vocabulary import SymbolicVocab
            self.symbolic_vocab = SymbolicVocab(args.objects_vocab, args.attributes_vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        
        datum = self.data[item]

        ques_id = datum['question_id']
        ques = datum['question']

        if self.raw_dataset.splits[0] == 'vizwiz_val0' or self.raw_dataset.splits[0] == 'vizwiz_val1':
            img_dir = 'val'
        else:
            img_dir = 'test'

        # redefine np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        img_feature = np.load(os.path.join('data/vizwiz/vizwiz_%s_feats'%(img_dir),  datum['image']+ '.npz'))
        
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
            label = datum['answer']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                try:
                    target[self.raw_dataset.ans2label[ans]] = score
                except:
                    pass
            return ques_id, feats, boxes, ques, tags, target
        else:
            return ques_id, feats, boxes, ques, tags


class VIZEvaluator:
    def __init__(self, dataset: VIZDataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['answer']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)
        

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


