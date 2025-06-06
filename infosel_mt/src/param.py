# coding=utf-8
# Copyleft 2019 project LXRT.
import sys
sys.path.append('/storage/xiay41/vqa-prompt/visualbert/unsupervised_visualbert/src')

import argparse
import random

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)

def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default='none')

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/vqa_test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    parser.add_argument("--config", dest='config', default='configs/vqa.json', type=str)
    parser.add_argument("--save_folder", dest='save_folder', default="test", type=str)

    # Invalid parameters just designed to accomodate sgg code
    parser.add_argument("--config-file", dest="config-file", default=None, type=str)
    parser.add_argument("--algorithm", dest="algorithm", default=None, type=str)
    parser.add_argument("--save_path", dest="save_path", default=None, type=str)
    parser.add_argument("--filename", dest="filename", default=None, type=str)
    parser.add_argument("--dataname", dest="dataname", default=None, type=str)
    parser.add_argument("--features", dest="features", default='vqa', type=str)
    parser.add_argument("--model", dest="model", default=None, type=str)
    parser.add_argument("--use_amount", default=100, type=int, help='Number of training data')

    parser.add_argument("--apply_adapter", dest="apply_adapter", action='store_const', default=False, const=True)
    parser.add_argument("--unlikely_loss", dest="unlikely_loss", action='store_const', default=False, const=True)
    parser.add_argument("--use_fusion", dest="use_fusion", action='store_const', default=False, const=True)
    parser.add_argument("--train_ensb", dest="train_ensb", action='store_const', default=False, const=True) 
    parser.add_argument("--cl_loss", dest="cl_loss", action='store_const', default=False, const=True) 
    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Added by harold. Allows additional parameters specified by the json file.
    import commentjson
    from attrdict import AttrDict
    from pprint import pprint
    if args.config is not None:
        with open(args.config) as f:
            config_json = commentjson.load(f)
        dict_args = vars(args)
        dict_args.update(config_json)  # Update with overwrite
        args = AttrDict(dict_args)

    import shutil
    import os
    output = args.output
    if not os.path.exists(output):
        os.mkdir(output)
    shutil.copyfile(args.config, os.path.join(output, os.path.basename(args.config)))


    from pprint import pprint
    pprint(args)

    # print("\n\n\n\n")
    # with open(args.config) as f:
    #     print(f.read())
    
    return args


args = parse_args()
