# coding=utf-8
# Copyleft 2019 project LXRT.
import datetime
print(datetime.datetime.now())
import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import h5py
import pandas as pd
import json

from param import args
from pretrain.qa_answer_table import load_lxmert_qa, load_lxmert_from_sgg_and_lxmert_pretrain, load_lxmert_from_pretrain_noqa
from tasks.infosel_mt_model import InfoSelModel
from tasks.infosel_mlp import MLP
from tasks.infosel_gqa_data import GQATorchDataset
from tasks.infosel_viz_data import VIZTorchDataset

from utils import load_lxmert_sgg

# DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VQA:
    def __init__(self):
        # Model
        if 'bert' in args.optim:
            self.model = InfoSelModel()
        else:
            print('mlp')
            self.model = MLP()

        total_params = sum(p.numel() for p in self.model.parameters()if p.requires_grad)
        print(f"Number of parameters: {total_params}")

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.get("load_lxmert_pretrain", None) is not None:
            load_lxmert_from_pretrain_noqa(args.load_lxmert_pretrain, self.model)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model)
        # Datasets
        print('args.test', args.test)
        # print('args.test', args.test != 'none')
        if args.test == 'none':
            # print('args.test11', args.test)
            if args.dataname == 'vizwiz':
                tset = VIZTorchDataset(train='train', args=args)
            else:
                tset = GQATorchDataset(train='train', args=args)
            self.tdata_loader = DataLoader(
                tset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                drop_last=False, pin_memory=True,
                collate_fn=lambda x: x
                )
            if args.dataname == 'vizwiz':
                vset = VIZTorchDataset(train='val', args=args)
            else:
                vset = GQATorchDataset(train='val', args=args)
            self.vdata_loader = DataLoader(
                vset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                drop_last=False, pin_memory=True,
                collate_fn=lambda x: x
                )
            
            # Loss and Optimizer
            self.bce_loss = nn.BCEWithLogitsLoss()
            # self.bce_loss = nn.CrossEntropyLoss()
            if 'bert' in args.optim:
                batch_per_epoch = len(tset)
                t_total = int(batch_per_epoch * args.epochs)
                print("BertAdam Total Iters: %d" % t_total)
                from lxrt.optimization import BertAdam
                self.optim = BertAdam(list(self.model.parameters()),
                                    lr=args.lr,
                                    warmup=0.1,
                                    t_total=t_total)
            else:
    
                self.optim = args.optimizer(self.model.mlp.parameters(), args.lr)
            
        
        
        # GPU options
        self.model = self.model.to(device)
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
            self.model.multi_gpu()

        
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def check_updated_layers(self, model, param_dict):
        # check updated layers
        if len(param_dict)==0:
            for idx, (pn, p) in enumerate(model.named_parameters()):
                param_dict.append(p.clone())
                if p is not None and p.requires_grad: 
                    print('layers require for grad', pn)
                    # print('layers idx', idx)
        else:
            for idx, (pn, p) in enumerate(model.named_parameters()):

                if not torch.equal(p, param_dict[idx]):
                    print('updated layer:', pn)
                param_dict[idx] = p.clone()
        
        return param_dict

    def train(self, tdataloader, vdata_loader):
        # dset, loader, evaluator = train_tuple
        # iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        ms_param_dict_init = []
        best_valid = 0.
        train_results = []
        report_every = args.get("report_every", 100)
        for epoch in range(args.epochs):
            self.model.train()
            for i, batch in enumerate(tdataloader):
                ques_id, feats, boxes, tags, sent, target, candidates = zip(*batch)
                self.optim.zero_grad()

                target = torch.tensor(target).to(device)
                logit = self.model(feats, boxes, sent, tags, candidates, args.features)
                # assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                train_results.append(pd.Series({"loss":loss.detach().mean().item()}))
            
                
                if i % report_every == 0 and i > 0:
                    print("Epoch: {}, Iter: {}/{}".format(epoch, i, len(tdataloader)))
                    print("    {}\n~~~~~~~~~~~~~~~~~~\n".format(pd.DataFrame(train_results[-report_every:]).mean()))

                if epoch == 0 and i < 2:
                    print('check layers updated in model selector:')
                    self.check_updated_layers(self.model, ms_param_dict_init)
            
             # Do Validation
            valid_score = self.evaluate(vdata_loader)
            if valid_score > best_valid and not args.get("special_test", False):
                best_valid = valid_score
                self.save("BEST")

            log_str = "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                        "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            if epoch >= 5:
                self.save("Epoch{}".format(epoch))
            print(log_str, end='')
            print(args.output)


        self.save("LAST")

    def evaluate(self, vdataloader, dump=None):
        """Evaluate all data in data_tuple."""
        self.model.eval()
        results = []
        score = []
    
        for i, batch in enumerate(tqdm(vdataloader)):
            
            ques_id, feats, boxes, tags, sent, targets, candidates = list(zip(*batch))#, target = zip(*batch)
            with torch.no_grad():
                
                logits = self.model(feats, boxes, sent, tags, candidates, args.features)
                _, labels = logits.max(1)
                
                if args.dataname == 'vizwiz' and args.test == 'test':
                    for label, cand, qid in zip(labels, candidates, ques_id):   
                        results.append({'image': qid, 'answer': cand[label]})
                    score.append(0)
                else:
                    for label, cand, qid in zip(labels, candidates, ques_id):   
                        results.append({'image': qid, 'answer': cand[label]})
                    targets = torch.tensor(targets).to(device)
                    for label, target in zip(labels, targets):
                        # print('label, target', label, target)
                        if args.dataname == 'gqa':
                            if target[label] == 1:
                                score.append(1)
                            else:
                                score.append(0)
                        elif args.dataname == 'vqa' or args.dataname == 'vizwiz':
                            score.append(target[label])

                    

        if args.dataname == 'vizwiz' and args.test == 'test':
            print('save result to:', 'mc_outs/a_vizwiz_train_per/' + (args.output).split('/')[-1] + '_'+ str(args.use_amount)+ '.json')
            json.dump(results, open('mc_outs/a_vizwiz_train_per/' + (args.output).split('/')[-1] + '_'+ str(args.use_amount)+ '.json', 'w'))
            return 0.
        elif dump is not None:
            print('save result to:', dump)
            json.dump(results, open(dump, 'w'))
            return sum(score)/len(score)
        else:
            return sum(score)/len(score)


    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        pretrained_dict = torch.load("%s.pth" % path, map_location=torch.device('cpu'))
        if (args.dataname == 'gqa' and args.test == 'none') or (args.dataname == 'vizwiz' and args.test == 'none'):
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            for k, v in pretrained_dict.items():
                if k.split('.')[0] != 'logit_fc':
                    # print(k)
                    pretrained_dict = {k: v}
            
            # 2. overwrite entries in the existing state dict
            # print("self.model:", model_dict)
            # print("pretrained_dict:", pretrained_dict)
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(pretrained_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test != 'none':
        print('args.test', args.test)
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            if args.dataname == 'vizwiz':
                test_set = VIZTorchDataset(train='val', args=args)
            else:
                test_set = GQATorchDataset(train='test', args=args)
            test_dataloader = DataLoader(
            test_set, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            drop_last=False, pin_memory=True,
            collate_fn=lambda x: x
            )
            result = vqa.evaluate(
                test_dataloader, 
                dump='results/gqa_mt.json'
            )
            print('ACC:', result)

        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                vqa.vdata_loader

            )
            print('ACC:', result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        
        vqa.train(vqa.tdata_loader, vqa.vdata_loader)
        if args.dataname == 'vizwiz':
            test_set = VIZTorchDataset(train='test', args=args)
        else:
            test_set = GQATorchDataset(train='test', args=args)
        test_dataloader = DataLoader(
            test_set, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            drop_last=False, pin_memory=True,
            collate_fn=lambda x: x
            )
        args.test = 'test'
        if args.dataname == 'vizwiz':
            result = vqa.evaluate(
                    test_dataloader, 
                    dump='results/viz_mc_mtmt_test.json'
                )
        else:
            result = vqa.evaluate(
                    test_dataloader, 
                    dump='results/gqa_mc_mtmt_test.json'
                )
            print('ACC:', result)

