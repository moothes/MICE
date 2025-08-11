import argparse
import importlib
import os

def parse_args(model_name):
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")
    parser.add_argument("model", type=str, default="MICE", help="Name of model")
    parser.add_argument("--anno_file", type=str, default="./data/tcga_uni.csv", help="Data directory to WSI features (extracted via CLAM)")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for reproducible experiment")
    parser.add_argument("--text_encoder", type=str, default="dmis-lab/biobert-base-cased-v1.2", help="pretrained text encoder")
    parser.add_argument("--stage", type=str, choices=["split", "pretrain", "finetune"], default="pretrain", help="define the training stage")

    # Model Parameters.
    parser.add_argument("--modal", type=str, default="path,gene,text", help="Specifies which modalities to use / collate function in dataloader.")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="Number of folds")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead", "SGD"], default="AdamW")
    parser.add_argument("--scheduler", type=str, choices=["exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--num_epoch", type=int, default=20, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--awl", action="store_false", dest="awl", help="Auto-weighting for losses")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--loss", type=str, default="nllsurv", help="slide-level classification loss function (default: ce)")
    
    # Model-specific Parameters
    model_specific_config = importlib.import_module('models.{}.network'.format(model_name)).custom_config
    
    ### Base arguments with customized values
    parser.set_defaults(**model_specific_config['base'])
    
    ### Customized arguments
    for k, v in model_specific_config['customized'].items():
        v['dest'] = k
        parser.add_argument('--' + k, **v)
    
    args = parser.parse_args()
    return args
