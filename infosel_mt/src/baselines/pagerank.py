from cgi import test
import numpy as np
import pandas as pd
import torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
import argparse
import os
import warnings
import json
warnings.filterwarnings('ignore')
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def bleurt_score(references, candidates):
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12', cache_dir='/storage/xiay41/').to(device)
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12', cache_dir='/storage/xiay41/')
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt').to(device)
        res = model(**inputs).logits.flatten().tolist()
    
    return res

def page_rank(M, p=1, y=None, max_iterations=10):

    l = M.shape[0]
    x = np.ones(l) / l

    M=M/M.sum(axis=0)
    if y is None:
        y = np.ones(l) / l
    for i in range(max_iterations):
        x = p * y + (1-p) * M.dot(x)

    return x

def get_viz_score(pred, gt):
    if pred in gt:
        if gt.count(pred) > 3:
            score = 1.0
        else:
            score = gt.count(pred)/3
    else:
        score = 0.0

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='gqa', help='choose from [gqa, vizwiz]')  
    args = parser.parse_args()
    print('args:', args)
    ## load dataset
    path = './src/baselines/' + args.dataname
    if args.dataname == 'gqa':
    # train_label = json.load(open('../../abvlmo_all_answers_gqa_val0.json', 'r'))    
        test_label = json.load(open('data/gqa/abvlmo_all_answers_gqa_val1.json', 'r'))    
        test_y = []
        for item in test_label:
            gt = list(item['gt_ans'].keys())[0]
            albef = item['albef']
            blip = item['blip']
            vlmo = item['vlmo']
            test_y.append([float(gt==albef), float(gt==blip), float(gt==vlmo)])

    elif args.dataname == 'vizwiz':
        test_label = json.load(open('data/vizwiz/abvlmo_all_answers_viz_val1.json', 'r'))
    
    if os.path.isfile(path + '/responses_bleurt_score_val.csv'):
        overall_acc = 0
        bleurt_similarities_df = pd.read_csv(path + '/responses_bleurt_score_val.csv', encoding='utf-8', index_col=False)
        for row1, (index, row2) in zip(test_label, bleurt_similarities_df.iterrows()):
            adjacency_matrix = np.array([[1, row2['albef_blip'], row2['albef_vlmo']],
                                        [row2['albef_blip'], 1, row2['blip_vlmo']],
                                        [row2['albef_vlmo'], row2['blip_vlmo'], 1]])

       
            preds = []
            page_rank_scores=page_rank(adjacency_matrix, p=0)
            # print(page_rank_scores)
            pred = np.argmax(page_rank_scores)
            if args.dataname == 'gqa':
                overall_acc += row2['acc_label'][pred]
            elif args.dataname == 'vizwiz':
                gt = row1['gt_ans']
                overall_acc += [get_viz_score(row1['albef'], gt), get_viz_score(row1['blip'], gt), get_viz_score(row1['vlmo'], gt)][pred]
                # overall_f1[idx] += [row1['chatgpt_f1'], row1['gpt3_f1'], row1['llama_f1']][pred]

        print('test acc: ', overall_acc/len(test_label))

    else:
        overall_acc = 0
        results = []
        bleurt_similarities = []
        albef = []
        blip = []
        vlmo = []
        for index, row in enumerate(test_label):
            # if index < 2:
                A = row['albef']
                albef.append(A)
                B = row['blip']
                blip.append(B)
                C = row['vlmo']
                vlmo.append(C)
                # print(A, B, C)
                bleurt_similarity = bleurt_score([A, A, B], [B, C, C])
                # print(bleurt_similarity)
                bleurt_similarities.append(bleurt_similarity)

                AB, AC, BC = bleurt_similarity
                
                adjacency_matrix = np.array([[1,AB,AC],
                                    [AB,1,BC],
                                    [AC,BC,1]])
                
                page_rank_scores = page_rank(adjacency_matrix, p=0)
                # print(page_rank_scores)
                pred = np.argmax(page_rank_scores)
                # print(pred)
                if args.dataname == 'gqa':
                    overall_acc += test_y[index][pred]
                elif args.dataname == 'vizwiz':
                    results.append({'image': row['question_id'], 'answer': [A, B, C][pred]})
                # overall_f1 += [row['chatgpt_f1'], row['gpt3_f1'], row['llama_f1']][pred]

        sim_df = pd.DataFrame(np.array(bleurt_similarities),
                    columns=['albef_blip', 'albef_vlmo', 'blip_vlmo'])
        sim_df['albef'] = albef
        sim_df['blip'] = blip
        sim_df['vlmo'] =  vlmo
        if args.dataname == 'gqa':
            sim_df['acc_label'] = test_y
        sim_df.to_csv(path + '/responses_bleurt_score_val.csv', encoding='utf-8', index=False)

        if args.dataname == 'gqa':
            print('test acc: ', overall_acc/len(test_label))
        
        elif args.dataname == 'vizwiz':
            json.dump(results, open(path + '/pagerank_vizwiz_val.json', 'w'))
            print('save results to: ', path)