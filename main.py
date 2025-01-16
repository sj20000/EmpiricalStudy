# https://github.com/yongduosui/cal

from datasets import get_dataset
from train_causal import train_causal_real, train_causal_real_citation
from train import train, train_real, train_real_citation
import torch
import scipy.sparse as sp
import numpy as np
import opts
import utils
import pdb
import warnings
warnings.filterwarnings('ignore')
import time
from torch_geometric.datasets import Planetoid


def main():
    cit=True # for citation data 
    if cit==True: 
        args = opts.parse_args()
        dataset = Planetoid(root='.', name=args.dataset)
        data = dataset[0]
        model_func = opts.get_model(args)
        if args.model in ["GIN","GCN", "GAT", "GraphSAGE"]:
            train_real_citation(dataset, model_func, args)
        else:
            train_causal_real_citation(dataset, model_func, args)


    else: 
        args = opts.parse_args()
        dataset_name, feat_str, _ = opts.create_n_filter_triples([args.dataset])[0]
        dataset = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
        model_func = opts.get_model(args)

        if args.model in ["GIN","GCN", "GAT", "GraphSAGE"]:
            train_real(dataset, model_func, args)
        else:
            train_causal_real(dataset, model_func, args)

    
if __name__ == '__main__':
    main()

    