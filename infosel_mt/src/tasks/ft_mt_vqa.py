   # coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import json
import torch
import torch.nn as nn
# import torch.utils.data.Subset as Subset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import h5py
import pandas as pd
import pickle
from param import args
from pretrain.qa_answer_table import load_lxmert_qa, load_lxmert_from_sgg_and_lxmert_pretrain, load_lxmert_from_pretrain_noqa
from tasks.mt_model import VQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
from tasks.vizwiz_data import VIZDataset, VIZTorchDataset, VIZEvaluator
from utils import load_lxmert_sgg

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    if args.dataname == 'vizwiz':
        dset = VIZDataset(args.dataname, splits)
        tset = VIZTorchDataset(dset, args)
        evaluator = VIZEvaluator(dset)
    else:
        
        dset = GQADataset(args.dataname, splits)
        tset = GQATorchDataset(dset, args)
        evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True,
        collate_fn=lambda x: x
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=False
        )
        if args.valid != "":
            valid_bsize = args.get("valid_batch_size", 8)
            self.valid_tuple = get_data_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)
        # print('self.train_tuple.dataset.num_answers', self.train_tuple.dataset.num_answers)
        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.get("load_lxmert_pretrain", None) is not None:
            load_lxmert_from_pretrain_noqa(args.load_lxmert_pretrain, self.model)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.to(device)
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
            self.model.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.bce_loss = nn.CrossEntropyLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        train_results = []
        report_every = args.get("report_every", 100)
        for epoch in range(args.epochs):
            self.model.train()
            quesid2ans = {}
            for i, batch in enumerate(loader):

                ques_id, feats, boxes, sent, tags, target = zip(*batch)
                self.optim.zero_grad()
                target = torch.stack(target).to(device)
                logit = self.model(feats, boxes, sent, tags)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                train_results.append(pd.Series({"loss":loss.detach().mean().item()}))

                score, label = logit.max(1)

                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
                # print('epoch:', epoch)
                if i % report_every == 0 and i > 0:
                    print("Epoch: {}, Iter: {}/{}".format(epoch, i, len(loader)))
                    print("    {}\n~~~~~~~~~~~~~~~~~~\n".format(pd.DataFrame(train_results[-report_every:]).mean()))

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid and not args.get("special_test", False):
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            if epoch >= 5:
                self.save("Epoch{}".format(epoch))
            print(log_str, end='')
            print(args.output)

        self.save("LAST")

    def get_feature(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        results = []
        for i, batch in enumerate(tqdm(loader)):
            
            _ = list(zip(*batch))
            ques_id, feats, boxes, sent, tags = _[:5]#, target = zip(*batch)
            with torch.no_grad():
                #target = torch.stack(target).cuda()
                logits = self.model(feats, boxes, sent, tags)

                vqa_features_file = open('mt_features/'+args.dataname + '/val/eval_features_' + str(i) +'.txt', 'wb')
                pickle.dump(logits, vqa_features_file)
                vqa_features_file.close()

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        results = []
        for i, batch in enumerate(tqdm(loader)):
            
            _ = list(zip(*batch))
            ques_id, feats, boxes, sent, tags = _[:5]#, target = zip(*batch)
            with torch.no_grad():
                #target = torch.stack(target).cuda()
                logits = self.model(feats, boxes, sent, tags)

                _, label = logits.max(1)
                for qid, l, logit in zip(ques_id, label.cpu().numpy(), logits):
                    ans = dset.label2ans[l]
                    quesid2ans[qid]= ans
                    # if args.dataname == 'vizwiz' and dump is not None: 
                    results.append({'image': qid, 'answer': ans})
            
        if dump is not None:
            print('save result to:', dump)
            json.dump(results, open(dump, 'w'))
            return quesid2ans
        else:
            return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        pretrained_dict = torch.load("%s.pth" % path, map_location=torch.device('cpu'))

        if (args.dataname == 'gqa' and args.test == 'none') or (args.dataname == 'vizwiz' and args.test == 'none') or args.test == 'get_feature':
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            for k, v in pretrained_dict.items():
                if k.split('.')[0] != 'logit_fc':
                    # print(k)
                    pretrained_dict = {k: v}
            
            # 2. overwrite entries in the existing state dict
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
                result=vqa.evaluate(
                        get_data_tuple('vizwiz_val0', bs=8,
                                    shuffle=False, drop_last=False), 
                                    dump='mc_outs/a_vizwiz_train_per/' + (args.output).split('/')[-1] + '_'+ str(args.use_amount)+ 'val0.json'
                        )
                result=vqa.evaluate(
                        get_data_tuple('vizwiz_val1', bs=8,
                                    shuffle=False, drop_last=False), 
                                    dump='mc_outs/a_vizwiz_train_per/' + (args.output).split('/')[-1] + '_'+ str(args.use_amount)+ 'val1.json'
                        )
            else:
                result=vqa.evaluate(
                    get_data_tuple('gqa_testdev', bs=8,
                                shuffle=False, drop_last=False),
                    # dump='ft_mt_outs/gqa_fintune_mt_test.json'
                )
            print(result)
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('gqa_testdev', bs=8,
                               shuffle=False, drop_last=False),
                )
            print(result)
        
        elif args.test == 'get_feature':
            print('get_feature')
            if args.dataname == 'vizwiz':
                vqa.get_feature(get_data_tuple('vizwiz_val1', bs=8,
                                    shuffle=False, drop_last=False)
                )
            elif args.dataname == 'gqa':
                print('gqa')
                vqa.get_feature(get_data_tuple('gqa_val1', bs=8,
                                shuffle=False, drop_last=False), 
                                #    dump='ft_mt_outs/gqa_fintune_mt_test.json',
                )

    else:
        
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            #print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
        
        args.test == 'test'
        if args.dataname == 'vizwiz':
            result = vqa.evaluate(
                    get_data_tuple('test', bs=8,
                                shuffle=False, drop_last=False), 
                                dump='mc_outs/a_vizwiz_train_per/' + (args.output).split('/')[-1] + '_'+ str(args.use_amount)+ '.json'
                    )
        else:
            result = vqa.evaluate(
                get_data_tuple('val', bs=8,
                               shuffle=False, drop_last=False), 
                            #    dump='ft_mt_outs/gqa_fintune_mt_test.json',
                )
            print(result)

        