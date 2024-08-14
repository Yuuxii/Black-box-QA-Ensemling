## Data Collection

### GQA

Download image feature from [LXMERT](https://github.com/airsplay/lxmert):

`mkdir -p data/vg_gqa_imgfeatwget` 
`wget https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip -P data/vg_gqa_imgfeat`
`unzip data/vg_gqa_imgfeat/vg_gqa_obj36.zip -d data && rm data/vg_gqa_imgfeat/vg_gqa_obj36.zip`
`wget https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -P data/vg_gqa_imgfeat`
`unzip data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -d data && rm data/vg_gqa_imgfeat/gqa_testdev_obj36.zip` 

convert the .tsv files to h5 using src/tools/convert_tsv_to_h5.py

### VizWiz

There are no available extracted image features, we follow the process procedures as  [LXMERT](https://github.com/airsplay/lxmert) which uses [Faster R-CNN Feature Extraction](https://github.com/airsplay/lxmert?tab=readme-ov-file#faster-r-cnn-feature-extraction), a PyTorch implementation can be found: https://github.com/MILVLG/bottom-up-attention.pytorch. The model used to extract the feature is [R101-k36](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?download=1).



## InfoSel-TT
**Mini-SDv2**
```
python main/infosel_tt_qa.py --dataname sq --use_amount 1000 --output_dir tt_outs/sq/1k  --lr 5e-5 
```

**Mini-NQ**

```
python main/infosel_tt_qa.py --dataname nq --use_amount 1000 --output_dir tt_outs/nqopen/1k  --lr 5e-5 
```

## FT-BERT

**Mini-SDv2**
```
python main/ft_sq.py --use_amount 1000 --dataname sq
```

**Mini-NQ**
```
python main/ft_nq.py --use_amount 1000 --dataname nq
```

## Baselines for Textual QA


**OLA**
```
python baselines/dcs_ola.py --dataname sq
```

**PageRank**
```
python baselines/pagerank.py --dataname sq
```

## InfoSel-MT

Download images features and the initial checkpoints `vqa.pth` from https://github.com/uclanlp/visualbert/tree/master/unsupervised_visualbert .

**Mini-GQA**

```
python src/tasks/infosel_mt_vqa.py --output ./snap/gqa_models --config ./configs/gqa.json --dataname gqa --load ./snap/vqa --features vqa 
```

**Mini-Viz**

```
python src/tasks/infosel_mt_vqa.py --output ./snap/vizwiz_models --config ./configs/vizwiz.json --dataname vizwiz --load ./snap/vqa --features vqa 
```

## FT-MT

**Mini-GQA**
```
python src/tasks/ft_mt_vqa.py --output ./snap/gqa_finetune  --config ./configs/gqa.json  --load snap/vqa --dataname gqa 
```

**Mini-Viz**
```
python src/tasks/ft_mt_vqa.py --output ./snap/vizwiz_finetune  --config ./configs/vizwiz.json  --load snap/vqa --dataname vizwiz 
```

## Baselines for VQA

**OLA:**
```
python src/baselines/dcs_ola.py --dataname gqa
```
**PageRank:**
```
python src/baselines/pagerank.py --dataname gqa
```




