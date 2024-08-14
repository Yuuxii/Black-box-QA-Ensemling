import numpy as np
import pandas as pd
import torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

def bleurt_score(references, candidates):
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
        res = model(**inputs).logits.flatten().tolist()
    
    return res


def page_rank(M, p=1, y=None, max_iterations=10):
    # M += np.eye(M.shape[0]) * np.max(M)
    l = M.shape[0]
    x = np.ones(l) / l
    # print(M)
    # print(M.sum(axis=0))
    M=M/M.sum(axis=0)
    if y is None:
        y = np.ones(l) / l
    for i in range(max_iterations):
        x = p * y + (1-p) * M.dot(x)

    # print(x)
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sq', help='choose from [sq, nq]')  
    args = parser.parse_args()
    print('args:', args)
    ## load dataset

    if args.dataset == 'sq':
        path = './datasets/squadv2/'
        test = pd.read_csv(path + 'sqv2_with_full_scores_dev.csv', encoding='utf-8', index_col=False)
    elif args.dataset == 'nq':
        path = './datasets/c_nq/'
        test = pd.read_csv(path +'cnq_open_with_full_scores_dev.csv', encoding='utf-8', index_col=False)

    
    
    if os.path.isfile(path + 'responses_bleurt_score_test.csv'):
        overall_f1 = [0 for i in range(11)]
        overall_em = [0 for i in range(11)]
        bleurt_similarities_df = pd.read_csv(path + 'responses_bleurt_score_test.csv', encoding='utf-8', index_col=False)
        for (index, row1), (_, row2) in zip(test.iterrows(), bleurt_similarities_df.iterrows()):
            adjacency_matrix = np.array([[1, row2['chatgpt_gpt3'], row2['chatgpt_llama']],
                                        [row2['chatgpt_gpt3'], 1, row2['gpt3_llama']],
                                        [row2['chatgpt_llama'], row2['gpt3_llama'], 1]])

            preds = []
            for idx, p in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
                page_rank_scores=page_rank(adjacency_matrix, p=p/10)
            # print(page_rank_scores)
                pred = np.argmax(page_rank_scores)
                overall_em[idx] += [row1['chatgpt_em'], row1['gpt3_em'], row1['llama_em']][pred]
                overall_f1[idx] += [row1['chatgpt_f1'], row1['gpt3_f1'], row1['llama_f1']][pred]

        print('test em:', [em/test.shape[0] for em in overall_em])
        print('test f1:', [f1/test.shape[0] for f1 in overall_f1])

    else:
        overall_f1 = 0
        overall_em = 0
        bleurt_similarities = []
        for index, row in test.iterrows():
            # if index < 2:
            A = row['chatgpt_response']
            B = row['gpt3_response']
            C = row['llama_response']
            # print(A, B, C)
            bleurt_similarity = bleurt_score([A, A, B], [B, C, C])
            # print(bleurt_similarity)
            bleurt_similarities.append(bleurt_similarity)

            AB, AC, BC = bleurt_similarity
            
            adjacency_matrix = np.array([[1,AB,AC],
                                [AB,1,BC],
                                [AC,BC,1]])
            
       
            page_rank_scores = page_rank(adjacency_matrix, p=1)
            # print(page_rank_scores)
            pred = np.argmax(page_rank_scores)
            # print(pred)
            
            overall_em += [row['chatgpt_em'], row['gpt3_em'], row['llama_em']][pred]
            overall_f1 += [row['chatgpt_f1'], row['gpt3_f1'], row['llama_f1']][pred]

        sim_df = pd.DataFrame(np.array(bleurt_similarities),
                    columns=['chatgpt_gpt3', 'chatgpt_llama', 'gpt3_llama'])
        sim_df.to_csv(path + 'responses_bleurt_score_test.csv', encoding='utf-8', index=False)

        print('test em: ', overall_em/test.shape[0])
        print('test f1: ', overall_f1/test.shape[0])