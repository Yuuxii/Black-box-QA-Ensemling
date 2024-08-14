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
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.

MSCOCO_IMGFEAT_ROOT = 'visual_data/vqa/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}
Split2ImgFeatPath = {
    'vqa_bert_pos_answers_on_valid': 'visual_data/vqa/vqa_mscoco_imgfeat/val2014_obj36.h5',
    'vqa_bert_false_answers_on_valid': 'visual_data/vqa/vqa_mscoco_imgfeat/val2014_obj36.h5',
    # 'train': 'data/vqa/vqa_mscoco_imgfeat/train2014_obj36.h5',
    'bert_train': 'visual_data/vqa/vqa_mscoco_imgfeat/train2014_obj36.h5',
    'bert_nominival': 'visual_data/vqa/vqa_mscoco_imgfeat/val2014_obj36.h5',
    'bert_minival': 'visual_data/vqa/vqa_mscoco_imgfeat/val2014_obj36.h5',
    'train': 'visual_data/vqa/vqa_mscoco_imgfeat/val2014_obj36.h5',
    'val': 'visual_data/vqa/vqa_mscoco_imgfeat/val2014_obj36.h5',
    "test": 'visual_data/vqa/vqa_mscoco_imgfeat/test2015_obj36.h5',
}

GQA_MSCOCO_IMGFEAT_ROOT = 'visual_data/gqa/vg_gqa_imgfeat/'

GQA_Split2ImgFeatPath = {
    'val': 'visual_data/gqa/vg_gqa_obj36.h5',
    'train': 'visual_data/gqa/vg_gqa_obj36.h5',
    'test': 'visual_data/gqa/gqa_testdev_obj36.h5',
   
}


class ConcateH5():
    def __init__(self, list_of_h5):
        self.list_of_h5 = list_of_h5
        self.len_of_h5 = [len(i) for i in list_of_h5]
        
    def __getitem__(self, index):
        for i in range(0, len(self.len_of_h5)):
            if index < self.len_of_h5[i]:
                return self.list_of_h5[i][index]
            else:
                index -= self.len_of_h5[i]
    
    def __len__(self):
        return sum(self.len_of_h5)

"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
mapping_rawdataset_name_to_json = {
    "train": "train",
    "nominival": "val",
    "minival": "val"
}

class GQATorchDataset(Dataset):
    def __init__(self, train,  args):
        super().__init__()
        
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        ## load data for infosel-mt

        if train == 'train':
            self.preds = json.load(open('data/gqa/abvlmo_all_answers_gqa_val0.json', 'r'))
            self.gt = json.load(open('data/gqa/gqa_val0.json', 'rb'))
        elif train == 'val':
            self.preds = json.load(open('data/gqa/abvlmo_all_answers_gqa_val1.json', 'r'))    
            self.gt = json.load(open('data/gqa/gqa_val1.json', 'rb'))
        elif train == 'test':
            self.preds = json.load(open('data/gqa/abvlmo_all_answers_gqa_test.json', 'r'))    
            self.gt = json.load(open('data/gqa//gqa_testdev.json', 'rb'))

  
        self.limit_to_symbolic_split = args.get("limit_to_symbolic_split", False)
        if self.limit_to_symbolic_split:
            dataDir = "/local/harold/ubert/bottom-up-attention/data/vg/"
            coco_ids = set()
            self.mapping_cocoid_to_imageid = {}
            with open(os.path.join(dataDir, 'image_data.json')) as f:
                metadata = json.load(f)
                for item in metadata:
                    if item['coco_id']:
                        coco_ids.add(int(item['coco_id']))
                        self.mapping_cocoid_to_imageid[int(item['coco_id'])] = item["image_id"]

            # from lib.data.vg_gqa import vg_gqa
            # self.vg_gqa = vg_gqa(None, split = "val" if self.raw_dataset.name == "minival" else "train", transforms=None, num_im=-1)

        self.custom_coco_data = args.get("custom_coco_data", False)
        self.use_h5_file = args.get("use_h5_file", False)
        if self.use_h5_file:
            if args.dataname == 'vqa':
                self.image_feature_dataset = ImageFeatureDataset.create([train], Split2ImgFeatPath, on_memory = args.get("on_memory", False))
            elif args.dataname == 'gqa':
                self.image_feature_dataset = ImageFeatureDataset.create([train], GQA_Split2ImgFeatPath, on_memory = args.get("on_memory", False))
                
            self.ids_to_index = self.image_feature_dataset.ids_to_index

            # Screen data
            used_data = []
            for datum in self.gt:
                if args.dataname == 'vqa':
                    if datum['image'].split('/')[-1].split('.')[0] in self.ids_to_index:
                        used_data.append(datum)
                elif args.dataname == 'gqa':
                    if datum['image'] in self.ids_to_index:
                        used_data.append(datum)
            # print(len(self.ids_to_index))
            # print(self.ids_to_index)
        else:
            # Loading detection features to img_data
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It ispliys saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if topk is None else topk
            if args.dataname == 'vqa':
                img_data = load_obj_tsv(
                    os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[train])),
                    topk=load_topk)
            elif args.dataname == 'gqa':
                img_data = load_obj_tsv(
                    os.path.join(GQA_MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[train])),
                    topk=load_topk)

            # Convert img list to dict
            self.imgid2img = {}
            for img_datum in img_data:
                self.imgid2img[img_datum['image']] = img_datum
            
            used_data = self.gt         

        used_data = used_data[::args.get("partial_dataset", 1)]
        self.data = used_data

        # Only kept the data with loaded image features
        print("Use %d data in torch dataset" % (len(self.data)))

        if args.get("add_tags", False):
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )
            from lxrt.symbolic_vocabulary import SymbolicVocab
            self.symbolic_vocab = SymbolicVocab(args.objects_vocab, args.attributes_vocab)

    def load_custom_h5(self, h5_file):
        h5_features = h5_file['features']
        h5_boxes = deepcopy(np.array(h5_file['boxes']))
        h5_objects_id = deepcopy(np.array(h5_file['objects_id']))
        h5_objects_conf = deepcopy(np.array(h5_file['objects_conf']))
        h5_attrs_id = deepcopy(np.array(h5_file['attrs_id']))
        h5_attrs_conf = deepcopy(np.array(h5_file['attrs_conf']))
        return h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf

    def __len__(self):
        return len(self.preds)

    def __getitem__(self, item: int):
        
        datum = self.data[item]
        
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


        if self.custom_coco_data:
            image_index = self.ids_to_index[img_id]
            obj_num = None
            feats = self.h5_features[image_index]
            boxes = self.h5_boxes[image_index]
            img_h = self.h5_wh[image_index][1]
            img_w = self.h5_wh[image_index][0]

            obj_confs = None
            attr_labels = None
            attr_confs = None

        elif self.use_h5_file:
            '''image_index = self.ids_to_index[img_id]
            obj_num = 36
            feats = self.h5_features[image_index]
            boxes = self.h5_boxes[image_index]
            img_h = self.h5_wh[image_index][1]
            img_w = self.h5_wh[image_index][0] '''
            image_index, obj_num, feats, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs = self.image_feature_dataset[img_id]
            # print(image_index, obj_num, len(feats[0]), boxes, img_h, img_w, obj_labels, attr_labels)            
        else:
            # Get image info
            img_info = self.imgid2img[img_id]
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
            assert obj_num == len(boxes) == len(feats)
            img_h, img_w = img_info['img_h'], img_info['img_w']


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
            # assert datum['question_id'] == albef['question_id'] == blip['question_id'].item() == vbert['question_id']
            if args.dataname == 'vqa':       
                if answer.count(albef) >= 4:
                    a_score = 1.
                else:
                    a_score = answer.count(albef)/len(answer)*3      

                if answer.count(blip) >= 4:
                    b_score = 1.
                else:
                    b_score = answer.count(blip)/len(answer)*3

                if answer.count(vbert) >= 4:
                    v_score = 1.
                else:
                    v_score = answer.count(vbert)/len(answer)*3
                 
                label = [a_score, b_score, v_score]
            
            elif args.dataname == 'gqa':
                gt_label = list(answer.keys())[0]
                label = [float(gt_label==albef), float(gt_label==blip), float(gt_label==vbert)]
            
            return datum['question_id'], feats, boxes, tags, ques, label, [albef, blip, vbert]
        else:
            return datum['question_id'], feats, boxes, tags, ques



