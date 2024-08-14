from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import json
import argparse
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
import collections
import torch
from transformers import AutoModelForQuestionAnswering
from tqdm.auto import tqdm
import ast
import pandas as pd
import random
torch.manual_seed(9595)
random.seed(9595)
np.random.seed(9595)
# from torch.optim import AdamW
# from accelerate import Accelerator
import os
# from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truths):
#   truths = ast.literal_eval(truths)
  return max([int(normalize_text(prediction) == normalize_text(truth)) for truth in truths])

def get_tokens(s):
  if not s: return []
  return normalize_text(s).split()

def compute_f1(prediction, truths):
 
  all_f1, all_prec, all_rec = [],[],[]
  for truth in truths:
    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(truth)
    common = collections.Counter(truth_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
      # print(pred_tokens, truth_tokens)
      score = int(pred_tokens == truth_tokens)
      return [score, score, score ]
    
      
    # if there are no common tokens then f1 = 0
    if num_same == 0:
      return [0, 0, 0]
    
    prec = num_same / len(pred_tokens)
    rec = num_same / len(truth_tokens)
    f1 = (2 * prec * rec) / (prec + rec)

    all_f1.append(f1)
    all_prec.append(prec)
    all_rec.append(rec)

  final_f1 = max(all_f1)
  final_rec = all_rec[all_f1.index(final_f1)]
  final_prec = all_prec[all_f1.index(final_f1)]
  
  # if 0<rec<1:
  #   rec = 0.0
  
  return final_f1, final_prec, final_rec

def return_print(em, prec, rec, f1):
  print(str(round(em*100, 2)) + ' & ' + str(round(prec*100, 2))+ ' & ' + str(round(rec*100, 2)) + ' & ' + str(round(f1*100, 2)))    

def avg(em):
  return sum(em)/len(em)

def compute_metrics(start_logits, end_logits, features, examples, args):
    n_best = 20
    max_answer_length = 30
    # metric = evaluate.load("squad")

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    ems = []
    precs = []
    recs = []
    f1s = []
    anses =[]
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # if offsets[start_index][0] == offsets[end_index][1]:
                    #     answer = {
                    #         "text": '',
                    #         "logit_score": start_logit[start_index] + end_logit[end_index],
                    #     }
                    # else:
                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answer= best_answer["text"]

        else:
            predicted_answer = ''

        if args.dataname == 'sq':
            gt_answers = example["answers"]['text']
        elif args.dataname == 'nq':
            gt_answers = example["label"]

        if len(gt_answers) == 0:
            gt_answers = ['']

        # print(predicted_answer,gt_answers )
        ems.append(compute_exact_match(predicted_answer, gt_answers))
        f1, prec, rec = compute_f1(predicted_answer, gt_answers)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        anses.append(predicted_answer)
    
    print('final for: ', args.use_amount)
    return_print(avg(ems), avg(precs), avg(recs), avg(f1s))
    if args.save_predictions:
       with open('predictions/'+'ft_'+ args.dataname+'_'+ str(args.use_amount)+ '_' + str(len(anses)) + '.json', 'w') as f:
            json.dump(anses, f)

def preprocess_training_examples(examples):
    max_length = 512
    stride = 128
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
            
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
   
    return inputs

    # Processing the validation data
def preprocess_validation_examples(examples):
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    max_length = 512
    stride = 128
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def nq_gen(data):

    for idx in data.index:
        if len(ast.literal_eval(data['answers'][idx])) == 0:
            answer = {'text': [''], 'answer_start': [0]}
            label = ['']
            
        else:
            answer = ast.literal_eval(data['answers'][idx])[0]
            label = ast.literal_eval(data['label'][idx])
            answer = {'text': [answer['text']], 'answer_start': [answer['char_spans'][0][0]]}
      
        yield {'id':data['qid'][idx],
        'context': data['context'][idx],
        'question': data['question'][idx],
        'answers': answer,
        'label': label
        }


def main(args):
    
    gen = nq_gen
    full_data = pd.read_csv('./datasets/c_nq/nq_open_train_10k.csv', encoding='utf-8',index_col=False)
    if args.use_amount >= 10000:
        full_data_plus = pd.read_csv('./datasets/c_nq/nq_open_train_10k_more.csv', encoding='utf-8',index_col=False)
        full_data = pd.concat([full_data, full_data_plus])   
        full_data = full_data.reset_index(drop=True)
    test_data = pd.read_csv('./datasets/c_nq/nq_open_dev.csv', encoding='utf-8',index_col=False)
    raw_test_data = Dataset.from_generator(generator=gen, gen_kwargs={'data': test_data})
    test_dataset = raw_test_data.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_test_data.column_names,
    )
    print('test data len', len(raw_test_data), len(test_dataset))

    train_data = full_data.iloc[:round(args.use_amount*0.8)]
    train_data = train_data.reset_index(drop=True)
    val_data = full_data.iloc[round(args.use_amount*0.8):args.use_amount]
    val_data = val_data.reset_index(drop=True)

    raw_train_data = Dataset.from_generator(generator=gen, gen_kwargs={'data': train_data})
    raw_val_data = Dataset.from_generator(generator=gen, gen_kwargs={'data': val_data})

    train_dataset = raw_train_data.map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_train_data.column_names,
    )

    print('train data len:', len(raw_train_data), len(train_dataset))

    validation_dataset = raw_val_data.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_val_data.column_names,
    )
    print('val data len: ', len(raw_val_data), len(validation_dataset))

    # Processing the training data
    if args.use_pre:
        model_checkpoint = "bert-base-cased"
    else:
        model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint).to(
        device
    )

    # # Fine tuning the model with the 'Trainer' API
    train_args = TrainingArguments(
        "./finetuned_bertmodel/bert-finetuned-nq_" + str(args.use_amount),
        overwrite_output_dir = 'True',
        evaluation_strategy="no",
        save_strategy="epoch",
        # load_best_model_at_end=True,
        learning_rate=5e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size=4,
        # save_total_limit = 2,
        seed=9595,
        # label_names = ["start_positions", "end_positions"],
        # fp16=True,
        # push_to_hub=True,
        report_to='none'
    )


    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    print('testing: ')
    predictions, _, _ = trainer.predict(test_dataset)
    start_logits, end_logits = predictions
    
    compute_metrics(start_logits, end_logits, test_dataset, raw_test_data, args)

    if args.save_predictions:
       train_dataset = raw_train_data.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_train_data.column_names,
        )
       predictions, _, _ = trainer.predict(train_dataset)
       start_logits, end_logits = predictions
       compute_metrics(start_logits, end_logits, train_dataset, raw_train_data, args)

       predictions, _, _ = trainer.predict(validation_dataset)
       start_logits, end_logits = predictions
       compute_metrics(start_logits, end_logits, validation_dataset, raw_val_data, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='sq', help='choose from [sq, nq]')    
    parser.add_argument('--use_amount', default=1000, type=int) 
    parser.add_argument('--use_pre', action='store_true') 
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--save_predictions', action='store_true')
    args = parser.parse_args()
    print('args:', args)
    main(args)