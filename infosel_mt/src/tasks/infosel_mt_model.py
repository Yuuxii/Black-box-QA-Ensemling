# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from param import args
from lxrt.infosel_entry import LXRTEncoder, convert_sents_to_features_tensors, convert_tags_to_tensorts, pad_np_arrays
from lxrt.infosel_modeling import BertLayerNorm, GeLU
from lxrt.tokenization import BertTokenizer
import numpy as np

# Max length including <bos> and <eos>
MAX_LENGTH = 40


class InfoSelModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            # nn.Linear(hid_dim* 3, hid_dim),
            
            # GeLU(),
            # BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 3, 3)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )
    
    def multi_gpu(self):
        self.lxrt_encoder.model.module.bert = nn.DataParallel(self.lxrt_encoder.model.module.bert)

    def forward(self, feat, pos, sent, tags, candidates, feats):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        #x = self.lxrt_encoder(sent, (feat, pos))
        # print(sent)
       
        input_ids, input_mask, segment_ids = convert_sents_to_features_tensors(sent, candidates, max_seq_length=MAX_LENGTH, tokenizer=self.tokenizer, feats=feats)

        # print('sent', input_ids.size(), input_mask.size(), segment_ids.size())
        visual_tags, visual_tags_mask, visual_tags_box, visual_tags_type, visual_tags_segment_ids = convert_tags_to_tensorts(tags)

        feat = pad_np_arrays(feat, padding_value=0, dtype=np.float32)
        pos = pad_np_arrays(pos, padding_value=0, dtype=np.float32)

 
        indiv_pooled_outputs, candidate_num  = self.lxrt_encoder.model.module.bert(
                input_ids, segment_ids, input_mask,
                visual_feats=(feat, pos),
                visual_attention_mask=None,
                visual_feats_seg_ids=None,
                visual_tags=visual_tags, 
                visual_tags_mask=visual_tags_mask,
                visual_tags_box=visual_tags_box, 
                visual_tags_type=visual_tags_type, 
                visual_tags_segment_ids=visual_tags_segment_ids,
                feats=feats
                )

        all_outs = torch.cat([indiv_pooled_outputs[i] for i in range(candidate_num)], dim=1)
        
        
        logit = self.logit_fc(all_outs)

        return logit


