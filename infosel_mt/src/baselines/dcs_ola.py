from unittest.util import _MAX_LENGTH
import warnings
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.utils import check_array
from pyod.utils.utility import check_parameter
import torch
import os
import random
import pickle
import json

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

    def fit_predict_f1(self, train_X, train_y, test_X, test_y, save_results=False):
        """Fit classifier.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """

    
        train_X = check_array(train_X)
   
        # build KDTree out of training subspace
        self.tree_ = KDTree(train_X)

  
        self.fitted_ = True

        y_predicted_score = []


        # For each test sample
        results = []
        for index, (test_x, t_y) in enumerate(zip(test_X, test_y)):
            test_x = check_array(test_x.reshape(1, -1))
        
            # Find neighbors for one test instance
            if index == 0:
                print(len(train_X), self.local_region_size)
            _, ind_arr = self.tree_.query(test_x, k=self.local_region_size)

         
            train_inds = ind_arr
           
            y_train_sample = np.array(train_y)[train_inds[0]]
        
            clf_performance = np.zeros([3, ]) ## 3 = num of estimators

            for j in range(3):
            
                clf_performance[j] = sum([y[j] for y in y_train_sample])/len(y_train_sample)

            # select the best clf. may get multiple results
            select_clf_inds = np.argwhere(
                clf_performance == np.amax(clf_performance)).ravel()

            # select the first element from all candidates
            best_clf_ind = select_clf_inds[-1]
    
            # make prediction and calculate the f1 score
            y_predicted_score.append(t_y[best_clf_ind])

            if save_results:
                results.append({'image': t_y[3], 'answer': y_predicted_score[index]})
        
        if not save_results:
            return sum(y_predicted_score)/len(y_predicted_score)
        else:
            json.dump(results, open('baseline_outs/'+ str(len(train_X)) +'_ola_viz_test.json', 'w'))


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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='gqa', help='choose from [gqa, vizwiz]')  
    args = parser.parse_args()
    print('args:', args)

    ## load dataset
    train_X = []
    test_X = []
    val_X =[]
    if args.dataname == 'gqa':
        train_path = 'mt_features/gqa/train/'
        val_path  = 'mt_features/gqa/val/'
        test_path = 'mt_features/gqa/test/'
    elif args.dataname == 'vizwiz':
        train_path = 'mt_features/vizwiz/train/'
        val_path  = 'mt_features/vizwiz/val/'
        test_path = 'mt_features/vizwiz/test/'
    train_list = os.listdir(train_path)
    val_list = os.listdir(val_path)
    test_list = os.listdir(test_path)

    train_list.sort()
    test_list.sort()

    for i in range(len(train_list)):    
        train_file = open(train_path + 'eval_features_'+ str(i) + '.txt', 'rb') 
        items = pickle.load(train_file)
        for item in items:
            train_X.append(item.numpy())
    
    for i in range(len(val_list)):    
        val_file = open(val_path + 'eval_features_'+ str(i) + '.txt', 'rb') 
        items = pickle.load(val_file)
        for item in items:
            val_X.append(item.numpy())

    for i in range(len(test_list)):
        test_file = open(test_path + 'eval_features_' + str(i) + '.txt', 'rb')
        items = pickle.load(test_file)
        for item in items:
            test_X.append(item.numpy())

    if args.dataname == 'gqa':
        train_label = json.load(open('data/gqa/abvlmo_all_answers_gqa_val0.json', 'r'))
        val_label = json.load(open('data/gqa/abvlmo_all_answers_gqa_val1.json', 'r'))    
        test_label = json.load(open('data/gqa/abvlmo_all_answers_gqa_test.json', 'r'))    
        train_y = []
        val_y = []
        test_y = []
        for item in train_label:
            gt = list(item['gt_ans'].keys())[0]
            albef = item['albef']
            blip = item['blip']
            vlmo = item['vlmo']
            train_y.append([float(gt==albef), float(gt==blip), float(gt==vlmo)])
        
        for item in val_label:
            gt = list(item['gt_ans'].keys())[0]
            albef = item['albef']
            blip = item['blip']
            vlmo = item['vlmo']
            val_y.append([float(gt==albef), float(gt==blip), float(gt==vlmo), item['question_id']])

        for item in test_label:
            gt = list(item['gt_ans'].keys())[0]
            albef = item['albef']
            blip = item['blip']
            vlmo = item['vlmo']
            test_y.append([float(gt==albef), float(gt==blip), float(gt==vlmo)])

    elif args.dataname == 'vizwiz':
        train_label = json.load(open('data/vizwiz/abvlmo_all_answers_viz_val0.json', 'r'))
        val_label = json.load(open('data/vizwiz/abvlmo_all_answers_viz_val1.json', 'r'))    
        test_label = json.load(open('data/vizwiz/abvlmo_all_answers_viz_test.json', 'r'))    
        train_y = []
        val_y = []
        test_y = []
        for item in train_label:
            gt = item['gt_ans']
            albef = item['albef']
            blip = item['blip']
            vlmo = item['vlmo']
            train_y.append([get_viz_score(albef, gt), get_viz_score(blip, gt), get_viz_score(vlmo, gt)])
        
        for item in val_label:
            gt = item['gt_ans']
            albef = item['albef']
            blip = item['blip']
            vlmo = item['vlmo']
            val_y.append([get_viz_score(albef, gt), get_viz_score(blip, gt), get_viz_score(vlmo, gt)])

        for item in test_label:
            albef = item['albef']
            blip = item['blip']
            vlmo = item['vlmo']
            test_y.append([albef, blip, vlmo, item['question_id']])
        
    print(len(train_X), len(train_X[0]), len(train_y))
    print(len(val_X), len(train_X[0]), len(train_y))
    print(len(test_X), len(test_X[0]), len(test_y))

    num_of_samples_list = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

    for num in num_of_samples_list:
        used_amount = round(num*len(train_X))
        sampled_train_X = train_X[: used_amount]
        sampled_train_Y = train_y[: used_amount]
        val_score = DCS_OLA().fit_predict_f1(sampled_train_X, sampled_train_Y, val_X, val_y)
        if args.dataname == 'gqa':
            test_score = DCS_OLA().fit_predict_f1(sampled_train_X, sampled_train_Y, test_X, test_y)
        elif args.dataname == 'vizwiz':
            test_score = DCS_OLA().fit_predict_f1(sampled_train_X, sampled_train_Y, test_X, test_y, save_results=True)
        print(num, val_score, test_score)


