from re import A
from xml.parsers.expat import model
from numpy.random.mtrand import sample
# from torch.autograd.grad_mode import F
import torch.nn as nn
import torch
import argparse
from infosel_qa_data import MCEnsbDataset
from torch import optim as optim
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import evaluate
# from huggingface_hub.hf_api import HfFolder
# from accelerate import load_checkpoint_and_dispatch, init_empty_weights
# from huggingface_hub import snapshot_download
import os
import pandas as pd
import random
torch.manual_seed(9595)
random.seed(9595)
np.random.seed(9595)

accuracy = evaluate.load("accuracy")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # num_items_in_batch = kwargs.get('num_items_in_batch')
        labels = inputs.get("labels")
        # forward pass
        # print('input', inputs)
        outputs = model(**inputs)
        # print('output', outputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = []

    for pred, label in zip(predictions, labels):
        score.append(label[pred])
 
    acc =  sum(score)/len(predictions)
    print('m1 f1 score:', sum(row[0] for row in labels)/len(labels))
    print('m2 f1 score:', sum(row[1] for row in labels)/len(labels))
    try:
        print('m3 f1 score:', sum(row[2] for row in labels)/len(labels))
    except:
        pass
    print('ensb f1 score:', acc)
    return {'accuracy': acc}

def return_print(em, prec, rec, f1):
  print('final_result:', args.use_amount)
  print(str(round(em*100, 2)) + ' & ' + str(round(prec*100, 2))+ ' & ' + str(round(rec*100, 2)) + ' & ' + str(round(f1*100, 2)))    

def test(trainer, test_dataset):
    print('---test---')
    predictions, labels, _ = trainer.predict(test_dataset)
    if args.dataname == 'sq':
        if os.path.isfile('./datasets/squadv2/sqv2_with_full_scores_dev_ensemble.csv'):
            data = pd.read_csv('./datasets/squadv2/sqv2_with_full_scores_dev_ensemble.csv', encoding='utf-8',
                                    index_col=False)
        else:
            data = pd.read_csv('./datasets/squadv2/sqv2_with_full_scores_dev.csv', encoding='utf-8',
                                    index_col=False)
    elif args.dataname == 'nq':
        if os.path.isfile('./datasets/c_nq/cnq_open_with_full_scores_dev_ensemble.csv'):
            data = pd.read_csv('./datasets/c_nq/cnq_open_with_full_scores_dev_ensemble.csv', encoding='utf-8',
                                    index_col=False)
        else:
            data = pd.read_csv('./datasets/c_nq/cnq_open_with_full_scores_dev.csv', encoding='utf-8',
                                    index_col=False)
        
    ems, precs, recs, anses  = [], [], [], []

    for index in data.index:
        if args.exclude_worst_model:
            ems.append([data['gpt3_em'][index], data['llama_em'][index]])
            precs.append([data['gpt3_prec'][index], data['llama_prec'][index]])
            recs.append([data['gpt3_prec'][index], data['llama_rec'][index]])
            anses.append([data['gpt3_response'][index], data['llama_response'][index]]) 
        else:
            ems.append([data['chatgpt_em'][index], data['gpt3_em'][index], data['llama_em'][index]])
            precs.append([data['chatgpt_prec'][index], data['gpt3_prec'][index], data['llama_prec'][index]])
            recs.append([data['chatgpt_rec'][index], data['gpt3_rec'][index], data['llama_rec'][index]])
            anses.append([data['chatgpt_response'][index], data['gpt3_response'][index], data['llama_response'][index]]) 
    # predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score, prec, rec, em, ans, preds = [], [], [], [], [], []

    for pred, label, e, p, r, a in zip(predictions, labels, ems, precs, recs, anses):
        score.append(label[pred])
        em.append(e[pred])
        prec.append(p[pred])
        rec.append(r[pred])
        ans.append(a[pred])
        preds.append(pred)
        # if label[pred] == 1:
        #     corrects += 1 
    # return accuracy.compute(predictions=predictions, references=labels)
    acc =  sum(score)/len(predictions)
    print('m1 f1 score:', sum(row[0] for row in labels)/len(labels))
    print('m2 f1 score:', sum(row[1] for row in labels)/len(labels))
    try:
        print('m3 f1 score:', sum(row[2] for row in labels)/len(labels))
    except:
        pass
    print('ensb f1 score:', acc)
    print('predictions count:', np.unique(predictions, return_counts=True))
    print('oracle em:', sum([max(em_values) for em_values in ems])/len(ems))
    print('oracle f1:', sum([max(label) for label in labels])/len(labels))
    return_print(sum(em)/len(predictions), sum(prec)/len(predictions), sum(rec)/len(predictions), acc)

    if not (args.exclude_worst_model or args.random_sample):
        data['ensb'+ str(args.use_amount) + '_ans'] = ans
        data['ensb'+ str(args.use_amount) + '_f1'] = score
        data['ensb'+ str(args.use_amount) + '_em'] = em
        data['ensb'+ str(args.use_amount) + '_prec'] = prec
        data['ensb'+ str(args.use_amount) + '_rec'] = rec
        data['ensb'+ str(args.use_amount) + '_pred'] = preds
        if args.dataname == 'sq':
            data.to_csv('./datasets/squadv2/sqv2_with_full_scores_dev_ensemble.csv', encoding='utf-8', index=False)
        elif args.dataname == 'nq':
            data.to_csv('./datasets/c_nq/cnq_open_with_full_scores_dev_ensemble.csv', encoding='utf-8', index=False)
    elif args.exclude_worst_model:
        if args.dataname == 'sq':
            data = pd.read_csv('./datasets/squadv2/sqv2_with_full_scores_dev_ensemble.csv', encoding='utf-8',
                                    index_col=False)
        elif args.dataname == 'nq':
            data = pd.read_csv('./datasets/c_nq/cnq_open_with_full_scores_dev_ensemble.csv', encoding='utf-8',
                                    index_col=False)
            
        data['ensb_exclude_gpt3'+ str(args.use_amount) + '_ans'] = ans
        data['ensb_exclude_gpt3'+ str(args.use_amount) + '_f1'] = score
        data['ensb_exclude_gpt3'+ str(args.use_amount) + '_em'] = em
        data['ensb_exclude_gpt3'+ str(args.use_amount) + '_prec'] = prec
        data['ensb_exclude_gpt3'+ str(args.use_amount) + '_rec'] = rec
        data['ensb_exclude_gpt3'+ str(args.use_amount) + '_pred'] = preds

        print('counts for models: ', preds.count(0)/len(preds), preds.count(1)/len(preds))
        print('counts for three models: ', data['ensb1000_pred'].tolist().count(0)/len(preds), data['ensb1000_pred'].tolist().count(1)/len(preds), data['ensb1000_pred'].tolist().count(2)/len(preds))

        if args.dataname == 'sq':
            data.to_csv('./datasets/squadv2/sqv2_with_full_scores_dev_ensemble.csv', encoding='utf-8', index=False)
        elif args.dataname == 'nq':
            data.to_csv('./datasets/c_nq/cnq_open_with_full_scores_dev_ensemble.csv', encoding='utf-8', index=False)

    return acc



def load_model():
    if args.use_pre:
            print('use pretrained model')
            model = AutoModelForMultipleChoice.from_pretrained("ncduy/bert-base-uncased-finetuned-swag")
            tokenizer = AutoTokenizer.from_pretrained("ncduy/bert-base-uncased-finetuned-swag")
    else:
        print('use bert base model')
        if args.model == 'bert':
            if args.evaluate:
                model = AutoModelForMultipleChoice.from_pretrained("mc_bert/nqopen/1per_seed_wt/checkpoint-400/")
                tokenizer = AutoTokenizer.from_pretrained("mc_bert/nqopen/1per_seed_wt/checkpoint-400/")
            else:
                model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        

    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true') 
    parser.add_argument('--use_pre',  action='store_true')  
    parser.add_argument('--eval_metric', default='f1', help='choose from [rec, f1]')    
    parser.add_argument('--include_model', default='gpt3', help='choose from [gpt3, flan]') 
    parser.add_argument('--model', default='bert', help='choose from [bert, roberta, albert]')   
    parser.add_argument('--output_dir', default='mc_bert/sqv2/10per', help='choose from [gpt3, flan]')  
    parser.add_argument('--dataname', default='sq', help='choose from [sq, nq]')  
    parser.add_argument('--num_gpus', default=1, type=int)  
    parser.add_argument('--use_amount', default=10000, type=int)  
    parser.add_argument('--bs', default=4, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--opt', default='adamw', help='choose from [sgd, adamw]')
    parser.add_argument('--test', action='store_true') # test on testdev data
    parser.add_argument('--shuffle', action='store_true') # test on testdev data
    parser.add_argument('--exclude_worst_model', action='store_true')
    parser.add_argument('--random_sample', action='store_true')
    parser.add_argument('--input_info', default='qa')
    args = parser.parse_args()
    print('args:', args)

    f1_scores = []
    if args.random_sample:
        sample_num = 10
    else:
        sample_num = 1

    for i in range(sample_num):
        # model and tokenizer
        model, tokenizer = load_model()
        
        ## data
        train_dataset = MCEnsbDataset(args, train='train',  tokenizer = tokenizer, random_state=i)
        eval_dataset = MCEnsbDataset(args, train='val', tokenizer = tokenizer, random_state=i)    
        # train_dataset, eval_dataset = torch.utils.data.random_split(train_eval_dataset, [0.8,  0.2])

        test_dataset = MCEnsbDataset(args, train='test', tokenizer = tokenizer)  

        if not args.evaluate:
            
            print('train_dataset', len(train_dataset)) 
    
        print('test_dataset', len(test_dataset))  

        print('eval_dataset', len(eval_dataset))

        save_strategy = 'epoch'
        load_best_model_at_end = True
        
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir = 'True',
            evaluation_strategy="epoch",
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            learning_rate=args.lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=5,
            weight_decay=0.01,
            save_total_limit = 2,
            seed=9595,
            report_to='none'
            )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            )

    
        if not args.evaluate:
            print('start training!')
    
            trainer.train()
        
            f1 = test(trainer, test_dataset)
            if not args.random_sample: 
                trainer.save_model(args.output_dir)
            f1_scores.append(f1)

            # trainer.predict(test_dataset)
            # trainer.predict(eval_dataset)
 
    if args.random_sample:
        print('f1 scores: ', f1_scores)
        print('avg: ', sum(f1_scores)/len(f1_scores))
        print('sd: ', torch.std(torch.tensor(f1_scores)))