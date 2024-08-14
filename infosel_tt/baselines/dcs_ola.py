
import encodings
from unittest.util import _MAX_LENGTH
import warnings
import numpy as np

from sklearn.neighbors import KDTree
# from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
# from sklearn.utils import check_X_y
# from sklearn.utils.validation import check_is_fitted
# from sklearn.utils.multiclass import check_classification_targets
import pandas as pd
from pyod.utils.utility import check_parameter
from tensorflow.python.framework.ops import no_gradient
from transformers import AutoTokenizer, BertTokenizer, BertModel
import torch
import os
import pandas as pd
import argparse
import random
torch.manual_seed(9595)
random.seed(9595)
np.random.seed(9595)


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DCS_OLA:
    """Dynamic Classifier Selection (DCS) is an established combination
    framework for classification tasks. The technique was first proposed by Ho
    et al. in 1994 :cite:`ho1994decision` and then extended, under the name
    DCS Local Accuracy, by Woods et al. in 1997 :cite:`woods1997combination`
    to select the most accurate base classifier in a local region.
    The motivation behind this approach is that base classifiers often make
    distinctive errors and over a degree of complementarity. Consequently,
    selectively combining base classifier can result in a performance
    improvement over generic ensembles which use the majority vote of all
    base classifiers.

    See :cite:`woods1997combination` for details.

    """

    def __init__(self, local_region_size=7, threshold=None,
                 pre_fitted=None):

        # validate input parameters
        if not isinstance(local_region_size, int):
            raise ValueError('local_region_size must be an integer variable')
        check_parameter(local_region_size, low=2, include_left=True,
                        param_name='local_region_size')
        self.local_region_size = local_region_size

        if threshold is not None:
            warnings.warn(
                "DCS does not support threshold setting option. "
                "Please set the threshold in classifiers directly.")

        if pre_fitted is not None:
            warnings.warn("DCS does not support pre_fitted option.")

    def fit_predict_f1(self, train_X, train_y, test):
        """Fit classifier.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """

        # Validate inputs X and y
    
        train_X = check_array(train_X)

        # build KDTree out of training subspace
        self.tree_ = KDTree(train_X)


        self.fitted_ = True

        
        n_samples = test.shape[0]
        y_predicted_f1 = np.zeros([n_samples, ])
        y_predicted_em = np.zeros([n_samples, ])
        # For each test sample
        for index, row in test.iterrows():
            test_x, test_em, test_f1 = get_data_embedding_test(model, tokenizer, row)
            test_x = check_array(test_x.cpu().detach().numpy())
        
            # Find neighbors for one test instance
            _, ind_arr = self.tree_.query(test_x, k=self.local_region_size)

            train_inds = ind_arr
            # print(train_inds)
            # ground truth
            y_train_sample = np.array(train_y)[train_inds[0]]
        
            clf_performance = np.zeros([3, ]) ## 3 = num of estimators

            for j in range(3):
               
                clf_performance[j] = sum([y[j] for y in y_train_sample])/len(y_train_sample)

            # select the best clf. may get multiple results
            select_clf_inds = np.argwhere(
                clf_performance == np.amax(clf_performance)).ravel()

            # select the first element from all candidates
            # print('select_clf_inds', select_clf_inds)
            best_clf_ind = select_clf_inds[-1]
            # print('best_clf_ind', best_clf_ind)
            # make prediction and calculate the f1 score
            y_predicted_em[index] = test_em[best_clf_ind]
            y_predicted_f1[index] = test_f1[best_clf_ind]


        return sum(y_predicted_em)/len(y_predicted_em), sum(y_predicted_f1)/len(y_predicted_f1)



def get_data_embedding_train(data, num_of_samples):
    data_X = []
    data_y = []
    for idx in data.index:
        if idx < num_of_samples:
            sentence = data['prompt'][idx]
            encoding = tokenizer(sentence, return_tensors="pt", max_length=512, padding='max_length', truncation=True).to(device)
            model.eval()
            with torch.no_grad():
                try:
                    output = model(**encoding)
                except:
                    print(sentence)
                mean_output = output.last_hidden_state.mean(dim=1)[0]
                # pooled_output = output.pooler_output
                if idx == 0:
                    print('embedding len: ', mean_output.size())
                data_X.append(mean_output.cpu().detach().numpy())

                data_y.append([data['chatgpt_f1'][idx], data['gpt3_f1'][idx], data['llama_f1'][idx]])

    return data_X, data_y

def get_data_embedding_test(model, tokenizer, instance):
    sentence = instance['prompt']
    encoding = tokenizer(sentence, return_tensors="pt", max_length=512, padding='max_length', truncation=True).to(device)
    model.eval()
    with torch.no_grad():
        try:
            output = model(**encoding)
        except:
            print(sentence)
        mean_output = output.last_hidden_state.mean(dim=1)

    return mean_output, [instance['chatgpt_em'], instance['gpt3_em'], instance['llama_em']], [instance['chatgpt_f1'], instance['gpt3_f1'], instance['llama_f1']]
    
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='sq', help='choose from [sq, nq]')  
    args = parser.parse_args()
    print('args:', args)

    ## load dataset
    if args.dataname == 'sq':
        # sq
        train = pd.read_csv('./datasets/squadv2/sqv2_with_full_scores_train_10k.csv', encoding='utf-8',
                                                index_col=False)
        test = pd.read_csv('./datasets/squadv2/sqv2_with_full_scores_dev.csv', encoding='utf-8',
                                        index_col=False)
    elif args.dataname == 'nq':
        #nq
        train = pd.read_csv('./datasets/c_nq/cnq_open_with_full_scores_train_10k.csv', encoding='utf-8',
                                                index_col=False)
        test = pd.read_csv('./datasets/c_nq/cnq_open_with_full_scores_dev.csv', encoding='utf-8',
                                        index_col=False)

    ## load model to process the the input
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    num_of_samples_list = [10, 50, 100, 500, 1000, 5000, 10000]
    for num in num_of_samples_list:
        train_X, train_y= get_data_embedding_train(train, num)
        # test_X, test_y = get_data_embedding(model, tokenizer, test)

        em_score, f1_score = DCS_OLA().fit_predict_f1(train_X, train_y, test)
        print(num)
        print(em_score, f1_score)


