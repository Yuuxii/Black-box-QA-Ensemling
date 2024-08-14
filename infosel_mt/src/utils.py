# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]




def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.size(0) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    
    
    logits = logits.squeeze(0)
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def top_p_filter(logits, threshold: float = 0.9):
    """
    Nucleus sampling
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - threshold)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k_logits(logits, k):
    '''
    Filters logits to only the top k choices
    from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
    '''
    if k == 0:
        return logits
    else:
        val, ind = torch.topk(logits, k)
        probs = torch.full_like(logits, float("-inf"))
        probs.scatter_(1, ind, val)
        return probs

def remove_tokens_after_eos(tensor, eos_token, image_token):
    # any tokens after and end of sequence token is produced are also set to the eos token, and removed
    
    # print(tensor, eos_token, (tensor == eos_token))
    eos_index = (tensor == eos_token).nonzero()
    if eos_index.any():
        tensor[eos_index[0] :] = eos_token

    tensor = tensor.tolist()
    return [i for i in tensor if (not i == image_token) and (not i == eos_token)]


def draw_fusion_plot(a, y_acc, dataname, filename=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(111, title='Fusion model')
    if dataname == 'vqa':
        ax0.hlines(y= 68.07, xmin = 0, xmax = max(a), color='red', linestyle ='dashed', linewidth = 2)
    elif dataname == 'gqa':
        ax0.hlines(y= 45.02, xmin = 0, xmax = max(a), color='red', linestyle ='dashed', linewidth = 2)
    ax0.plot(a, y_acc, 'bo-', label='fusion')
    
    if len(a) == 1:
        ax0.legend()
        
    fig.savefig(filename + '_fusion_.jpg')

def draw_loss_plot(x_epoch, y_loss, y_ulloss=None, y_acc=None, filename=None):
    fig = plt.figure()
    ax0 = fig.add_subplot(131, title="train loss vs valid loss")
    if y_ulloss != None:
        ax1 = fig.add_subplot(132, title=" ul loss")
    if y_acc != None:
        ax2 = fig.add_subplot(133, title="pos acc vs neg acc)")
    # y_loss[0] is train loss, y_loss[1] is val loss
    ax0.plot(x_epoch, y_loss[0], 'bo-', label='train')
    if y_loss != 0:
        ax0.plot(x_epoch, y_loss[1], 'ro-', label='val')
    # y_loss[0] is pos acc, y_loss[1] is neg acc
    if y_ulloss != None:
        ax1.plot(x_epoch, y_ulloss[0], 'bo-', label='train')
        if y_ulloss[1] != 0:
            ax1.plot(x_epoch, y_ulloss[1], 'ro-', label='val')
    if y_acc != None:
        ax2.plot(x_epoch, y_acc[0], 'bo-', label='pos acc')
        if y_acc[1] != 0:
            ax2.plot(x_epoch, y_acc[1], 'ro-', label='neg acc')
    if len(x_epoch) == 1:
        ax0.legend()
        if y_ulloss != None:
            ax1.legend()
        if y_acc != None:
            ax2.legend()
    fig.savefig(filename + '_.jpg')


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(tqdm(reader)):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

def load_obj_tsv_save_to_h5(fname, save_h5_name, save_json_name, all_examples):
    import h5py
    import json
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)

    metadata = []
    
    import h5py
    h5_file = h5py.File(save_h5_name, 'w')
    h5_features = h5_file.create_dataset('features', (all_examples, 36, 2048), dtype=np.float32)
    h5_boxes = h5_file.create_dataset('boxes', (all_examples, 36, 4), dtype=np.float32)
    h5_objects_id = h5_file.create_dataset('objects_id', (all_examples,36), dtype=np.int64)
    h5_objects_conf = h5_file.create_dataset('objects_conf', (all_examples,36), dtype=np.float32)
    h5_attrs_id = h5_file.create_dataset('attrs_id', (all_examples,36), dtype=np.int64)
    h5_attrs_conf = h5_file.create_dataset('attrs_conf', (all_examples,36), dtype=np.float32)

    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(tqdm(reader)):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            metadata.append(
                {
                    "img_id": item["img_id"],
                    "img_h": item["img_h"],
                    "img_w": item['img_w']
                }
            )
            h5_features[i] = item["features"]
            h5_boxes[i] = item["boxes"]
            h5_objects_id[i] = item["objects_id"]
            h5_objects_conf[i] = item["objects_conf"]
            h5_attrs_id[i] = item["attrs_id"]
            h5_attrs_conf[i] = item["attrs_conf"]


    with open(save_json_name, "w") as f:
        json.dump(metadata, f)
    return data

def create_slim_h5(fname, save_h5_name, save_json_name, all_examples, img_ids_to_keep):
    import h5py
    import json
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)

    metadata = []
    
    import h5py
    h5_file = h5py.File(save_h5_name, 'w')
    h5_features = h5_file.create_dataset('features', (all_examples, 36, 2048), dtype=np.float32)
    h5_boxes = h5_file.create_dataset('boxes', (all_examples, 36, 4), dtype=np.float32)
    h5_objects_id = h5_file.create_dataset('objects_id', (all_examples,36), dtype=np.int64)
    h5_objects_conf = h5_file.create_dataset('objects_conf', (all_examples,36), dtype=np.float32)
    h5_attrs_id = h5_file.create_dataset('attrs_id', (all_examples,36), dtype=np.int64)
    h5_attrs_conf = h5_file.create_dataset('attrs_conf', (all_examples,36), dtype=np.float32)
    i = 0
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for index, item in enumerate(tqdm(reader)):
            #continue
            if item["img_id"] not in img_ids_to_keep:
                continue

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            metadata.append(
                {
                    "img_id": item["img_id"],
                    "img_h": item["img_h"],
                    "img_w": item['img_w']
                }
            )
            h5_features[i] = item["features"]
            h5_boxes[i] = item["boxes"]
            h5_objects_id[i] = item["objects_id"]
            h5_objects_conf[i] = item["objects_conf"]
            h5_attrs_id[i] = item["attrs_id"]
            h5_attrs_conf[i] = item["attrs_conf"]
            i += 1
    with open(save_json_name, "w") as f:
        json.dump(metadata, f)
    return data

def load_lxmert_sgg(path, model):
    print("Load rel pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load(path)["model"]
    model_state_dict = model.state_dict()
    '''print("loaded_state_dict", loaded_state_dict["model"].keys())
    print("\n\n\n\n\n")
    print("model_state_dict", model_state_dict.keys())
    assert(0)'''

    new_loaded_state_dict = {}
    for key in list(loaded_state_dict.keys()):
        if "lxrt" in key:
            new_loaded_state_dict[key.split("lxrt.")[-1]] = loaded_state_dict[key]
            # module.rel_heads.rel_predictor.lxrt.encoder.r_layers.3.output.LayerNorm.weight -> encoder.r_layers.3.output.LayerNorm.weight

    load_state_dict_flexible(model.lxrt_encoder.model.bert, new_loaded_state_dict)
    
def load_lxmert_sgg_pretrain(path, model):
    print("Load rel pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load(path)["model"]
    model_state_dict = model.state_dict()
    '''print("loaded_state_dict", loaded_state_dict.keys())
    print("\n\n\n\n\n")
    print("model_state_dict", model_state_dict.keys())
    assert(0)'''

    new_loaded_state_dict = {}
    for key in list(loaded_state_dict.keys()):
        if "lxrt" in key:
            new_loaded_state_dict[key.split("lxrt.")[-1]] = loaded_state_dict[key]
            # module.rel_heads.rel_predictor.lxrt.encoder.r_layers.3.output.LayerNorm.weight -> encoder.r_layers.3.output.LayerNorm.weight

    load_state_dict_flexible(model.bert, new_loaded_state_dict)
    
def load_lxmert_to_sgg(path, model):
    print("Load rel pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load(path)["model"]
    model_state_dict = model.state_dict()
    '''print("loaded_state_dict", loaded_state_dict.keys())
    print("\n\n\n\n\n")
    print("model_state_dict", model_state_dict.keys())
    assert(0)'''

    new_loaded_state_dict = {}
    for key in list(loaded_state_dict.keys()):
        if "lxrt" in key:
            new_loaded_state_dict[key.split("lxrt.")[-1]] = loaded_state_dict[key]
            # module.rel_heads.rel_predictor.lxrt.encoder.r_layers.3.output.LayerNorm.weight -> encoder.r_layers.3.output.LayerNorm.weight

    load_state_dict_flexible(model.bert, new_loaded_state_dict)
    

def load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        print("Full loading failed!! Try partial loading!!")

    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print("Skipped: " + name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print("Successfully loaded: "+name)
        except:
            print("Part load failed: " + name)