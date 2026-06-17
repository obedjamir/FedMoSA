import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import random

from flcore.trainmodel.models import *

warnings.simplefilter("ignore")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(32)

def run(args):
    model_str = args.model
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: [{i+1}th/{args.times}] =============", flush=True)
        print("Creating server and clients ...")

        # Generate args.model
        if model_str == "mosa":
            args.model = MoSA(modalities=args.mod_list).to(args.device)
        elif model_str == "msavanilla":
            args.model = MSAVanilla().to(args.device)
        elif model_str == "nnunet":
            args.model = None
        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm.startswith("FednnUNET"):
            from flcore.servers.servernnUNET import FednnUNET
            server = FednnUNET(args, i)
        elif args.algorithm.startswith("FedMSA"):
            from flcore.servers.serverSAM import FedSAM
            server = FedSAM(args, i)
        elif args.algorithm.startswith("FedMoSA"):
            from flcore.servers.serverSAM import FedSAM
            server = FedSAM(args, i)
        else:
            raise NotImplementedError

        server.train()

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="experiment",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="liver_seg",
                        choices=["liver_seg", "brain_seg"])
    parser.add_argument('-m', "--model", type=str, default="mosa")
    parser.add_argument('-mn', "--model_name", type=str, default="mosa")
    parser.add_argument('-lbs', "--batch_size", type=int, default=8)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.0001,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=5)
    parser.add_argument('-ls', "--local_steps", type=int, default=2)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedMoSA",
                        choices=["FednnUNET", "FedMSA", "FedMoSA"])
    parser.add_argument('-jr', "--join_ratio", type=float, default=1,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-pr', "--prev_round", type=int, default=0,
                        help="Previous Global Round")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")

    # FL algorithms (multiple algs)
    parser.add_argument('-lam_bce', "--lam_bce", type=float, default=0.5)
    parser.add_argument('-lam_dice', "--lam_dice", type=float, default=0.5)
    parser.add_argument('-lam_proto', "--lam_proto", type=float, default=0.01)
    parser.add_argument('-afa', "--afa", type=int, default=0)
    parser.add_argument('-mod', '--mod_list', type=str, nargs='+', default=["MR", "CT"]) #Liver: ["MR","CT"], BraTS: ["T2", "FLAIR"]

    # save directories
    parser.add_argument("--hist_dir", type=str, default="../results/", help="dir path for output hist file")
    parser.add_argument("--log_dir", type=str, default="../logs/", help="dir path for log (main results) file")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoints/", help="dir path for checkpoints")

    #GPU
    parser.add_argument('-gpu', "--gpu_index", type=int, default=0)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #total_start = time.time()
    #time.sleep(5400)
    args = get_args()
    torch.cuda.set_device(args.gpu_index)

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    print("=" * 50)

    run(args)
