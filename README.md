# Pytorch Implementation of BIRD

SIGIR 2019: Finding Camouflaged Needle in a Haystack? Pornographic Products Detection via Berrypicking Tree Model

## Requirements

- torch
- numpy
- keras

I'm trying my best to make it lightweight.

## The Structure of the Folder

```
BIRD_torch
├── Dataset
│   ├── PPDD
│   │   ├── README
│   │   ├── data_list.txt    
│   │   ├── online_test_1.txt
│   │   ├── online_test_2.txt
│   │   ├── test_list.txt
│   │   ├── train_list.txt
│   │   └── val_list.txt
│   ├── dictionary
│   │   └── PPDD
│   │       └── word_index.json
│   └── sample
│       └── sample.txt
├── README.md
├── main.py 
├── model
│   ├── BIRD.py
│   ├── BPTRU.py
│   └── Network.py
└── utils
    ├── __init__.py
    └── data_loader.py
```

## Download the dataset

Follow the instrument in [https://github.com/GuoxiuHe/BIRD](https://github.com/GuoxiuHe/BIRD), and make sure the "Dataset" fold is in the location mentioned above.

## Run main.py

`main.py` is a simple train and test program with no gpu. You can run it with no arguments.

## Cite

If you use the codes or datasets, please cite the following paper:

```
@inproceedings{he2019finding,
  title={Finding Camouflaged Needle in a Haystack?: Pornographic Products Detection via Berrypicking Tree Model},
  author={He, Guoxiu and Kang, Yangyang and Gao, Zhe and Jiang, Zhuoren and Sun, Changlong and Liu, Xiaozhong and Lu, Wei and Zhang, Qiong and Si, Luo},
  booktitle={Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={365--374},
  year={2019},
  organization={ACM}
}
```