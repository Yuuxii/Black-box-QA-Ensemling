from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MCEnsbDataset(Dataset):
    def __init__(self, args, train, tokenizer, random_state=None):
        self.include_model = args.include_model
        self.eval_metric = args.eval_metric
        self.dataname = args.dataname
        self.input_info = args.input_info
        self.exclude_worst_model = args.exclude_worst_model
        self.random_sample = args.random_sample
        if self.dataname == 'sq':
            if train == 'test': 
                self.data = pd.read_csv('./datasets/squadv2/sqv2_with_full_scores_dev.csv', encoding='utf-8',
                                        index_col=False)
            else:
         
                self.data = pd.read_csv('./datasets/squadv2/sqv2_with_full_scores_train_10k.csv', encoding='utf-8',
                                        index_col=False)
                
                data_plus = pd.read_csv('./datasets/squadv2/sqv2_with_full_scores_train_10k_more.csv', encoding='utf-8',
                                    index_col=False)
                self.data = pd.concat([self.data, data_plus])   
                self.data = self.data.reset_index(drop=True) 

                if self.random_sample:
                    self.data = self.data.sample(n=args.use_amount, ignore_index=True, random_state=random_state)

                if train == 'train':  
                    self.data = self.data.iloc[0:round(args.use_amount*0.8)]
                elif train == 'val':
                    self.data = self.data.iloc[round(args.use_amount*0.8):round(args.use_amount*0.8)+round(args.use_amount*0.2)]
                    self.data=  self.data.reset_index(drop=True)

        elif self.dataname == 'nq':
            if train == 'test': 
                self.data = pd.read_csv('./datasets/c_nq/cnq_open_with_full_scores_dev.csv', encoding='utf-8',
                                        index_col=False)
            else:
                    
                self.data = pd.read_csv('./datasets/c_nq/cnq_open_with_full_scores_train_10k.csv', encoding='utf-8',
                                        index_col=False)
                
                
                data_plus = pd.read_csv('./datasets/c_nq/cnq_open_with_full_scores_train_10k_more.csv', encoding='utf-8',
                                        index_col=False)
                self.data = pd.concat([self.data, data_plus])   
                self.data = self.data.reset_index(drop=True)
                
                if self.random_sample:
                    self.data = self.data.sample(n=args.use_amount, ignore_index=True, random_state=random_state)
                
                if train == 'train':  
                    self.data = self.data.iloc[0:round(args.use_amount*0.8)]
                elif train == 'val':
                    self.data = self.data.iloc[round(args.use_amount*0.8):round(args.use_amount*0.8)+round(args.use_amount*0.2)]
                    self.data=  self.data.reset_index(drop=True)

        self.tokenizer = tokenizer

        # print('self.data', len(self.data))
    def __len__(self):
        
        return len(self.data.index.tolist())
    
    def __getitem__(self, index): 
        # if self.dataname == 'sq':
        #     question = self.data['prompt'][index]
        # elif self.dataname == 'nq':
        question = self.data['prompt'][index]

        chatgpt = self.data['chatgpt_response'][index]
        gpt3 = self.data['gpt3_response'][index]
        llama = self.data['llama_response'][index]
        
        if self.exclude_worst_model:
            anses = [gpt3, llama]

        else:
            anses = [chatgpt, gpt3, llama]

        ## use f1 as the optimization goal
        chatgpt_f1 = self.data['chatgpt_f1'][index]
        gpt3_f1 = self.data['gpt3_f1'][index]
        llama_f1 = self.data['llama_f1'][index]
        if self.exclude_worst_model:
            gt_ans = [gpt3_f1, llama_f1]
        else:
            gt_ans = [chatgpt_f1, gpt3_f1, llama_f1]

        # print('gt_ans:', gt_ans)
        num_ans = len(gt_ans)
        first_sentences = [[question] * num_ans ]
        second_sentences = [[f"{end}" for end in anses]]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        # print('error item: ', index, first_sentences, second_sentences)  
        tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True)

  
        if self.exclude_worst_model:
            return_item = {'question': question, 
                    'ans1':     anses[0], 
                    'ans2':     anses[1], 
                    'label':    gt_ans,
            }
        else:
            if self.input_info == 'q':
                return_item = {'question': question, 
                        'ans1':     '', 
                        'ans2':     '',
                        'ans3':     '', 
                        'label':    gt_ans,
                        # 'em': [self.data['chatgpt_em'][index],self.data['gpt3_em'][index], self.data['llama_em'][index] ],
                        # 'prec': [self.data['chatgpt_prec'][index],self.data['gpt3_prec'][index], self.data['llama_prec'][index] ],
                        # 'rec': [self.data['chatgpt_rec'][index],self.data['gpt3_rec'][index], self.data['llama_rec'][index] ],
                }
                
            elif self.input_info == 'a':
                return_item = {'question': '', 
                        'ans1':     anses[0], 
                        'ans2':     anses[1], 
                        'ans3':     anses[2],  
                        'label':    gt_ans,
                        # 'em': [self.data['chatgpt_em'][index],self.data['gpt3_em'][index], self.data['llama_em'][index] ],
                        # 'prec': [self.data['chatgpt_prec'][index],self.data['gpt3_prec'][index], self.data['llama_prec'][index] ],
                        # 'rec': [self.data['chatgpt_rec'][index],self.data['gpt3_rec'][index], self.data['llama_rec'][index] ],
                }
            else:
                return_item = {'question': question, 
                        'ans1':     anses[0], 
                        'ans2':     anses[1], 
                        'ans3':     anses[2],  
                        'label':    gt_ans,
                        # 'em': [self.data['chatgpt_em'][index],self.data['gpt3_em'][index], self.data['llama_em'][index] ],
                        # 'prec': [self.data['chatgpt_prec'][index],self.data['gpt3_prec'][index], self.data['llama_prec'][index] ],
                        # 'rec': [self.data['chatgpt_rec'][index],self.data['gpt3_rec'][index], self.data['llama_rec'][index] ],
                }

        for k, v in tokenized_examples.items():
            for i in range(0, len(v), num_ans):
                return_item[k] = v[i : i + num_ans]
                
        return return_item
                