import argparse
import torch
import numpy as np

import json
import yaml
import os

from main_model import CSDI_Physio
from dataset import get_dataloader
from utils import train, window_trick_evaluate_middle, evaluate_stable_loss

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:3', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=30)
parser.add_argument("--ratio",type=float,default=0.7)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--diffusion_step",type=int,default=50)
parser.add_argument("--machine_number",type=int,default=1)
parser.add_argument("--file",type=str)
parser.add_argument('--dataset',type=str,default="SMD")

parser.add_argument('--evaluate_stable', action="store_true", help="评估稳定分量损失")

args = parser.parse_args()


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)



# 由于是分开进行预测，
machine_number = args.machine_number

train_data_path_list = []
test_data_path_list = []
label_data_path_list = []


try:
    os.mkdir("window_result")
except:
    pass

for iteration in os.listdir("train_result"):

    try:
        os.mkdir(f"window_result/{iteration}")
    except:
        pass

    for subset_name in os.listdir(f"train_result/{iteration}/"):

        data_id = subset_name.split("_unconditional")[0]

        if "unconditional:True" in subset_name:
            unconditional = True
        else:
            unconditional = False

        split = 4
        diffusion_step = int(subset_name.split("diffusion_step:")[-1])

        train_data_path_list = []
        test_data_path_list = []
        label_data_path_list = []


        data_file = f"{data_id}_train.npy"
       
        if args.dataset == "SMD":
            train_data_path_list.append("data/Machine/SMD_train.npy")
            test_data_path_list.append("data/Machine/SMD_test.npy")
            label_data_path_list.append("data/Machine/SMD_test_label.npy")
       
        else:
           
            train_data_path_list.append(f"data/Machine/{args.dataset}_train.npy")
            test_data_path_list.append(f"data/Machine/{args.dataset}_test.npy")
            label_data_path_list.append(f"data/Machine/{args.dataset}_test_label.npy")


        # epoch = file.split("-")[0]
        train_data_path = train_data_path_list[0]
        test_data_path = test_data_path_list[0]
        label_data_path = label_data_path_list[0]
        train_loader, valid_loader, train_error_loader_list, test_loader_list = get_dataloader(
            train_data_path,
            test_data_path,
            label_data_path,
            batch_size=24,
            window_split=2,
            split=split
        )
        config["model"]["is_unconditional"] = unconditional
        config["model"]["test_missing_ratio"] = args.testmissingratio

        config["diffusion"]["num_steps"] = diffusion_step
        config["train"]["epochs"] = args.epochs
        print(json.dumps(config, indent=4))

        if args.dataset == "SMD":
            feature_dim = 33
        elif args.dataset == "PSM":
            feature_dim = 25
        elif args.dataset == "MSL":
            feature_dim = 55
        elif args.dataset == "SMAP":
            feature_dim = 25
        elif args.dataset == "GCP":
            feature_dim = 19
        elif args.dataset == "SWaT":
            feature_dim = 25

        model = CSDI_Physio(config, args.device, target_dim=feature_dim, ratio=args.ratio).to(args.device)
        base_folder = f"train_result/{iteration}/{subset_name}"

        model.load_state_dict(torch.load(f"{base_folder}/best-model.pth",map_location=args.device))

        print("base folder is ")
        print(base_folder)

        try:
            os.mkdir(f"window_result/{iteration}/{diffusion_step}")
        except:
            pass

        target_folder = f"window_result/{iteration}/{diffusion_step}/{subset_name}"

        try:
            os.mkdir(target_folder)
        except:
            continue

        # 评估稳定分量损失
        if args.evaluate_stable:
            print("\n开始评估稳定分量损失...")
            
            # 训练集稳定分量损失
            train_stable_loss = evaluate_stable_loss(
                model=model,
                dataloader=train_loader,
                foldername=target_folder,
                name=f"{data_id}_train"
            )
            
            # 测试集稳定分量损失
            test_stable_loss = evaluate_stable_loss(
                model=model,
                dataloader=test_loader_list[0],
                foldername=target_folder,
                name=f"{data_id}_test"
            )
            
            # 保存评估结果
            with open(f"{target_folder}/stable_loss_summary.json", "w") as f:
                json.dump({
                    "train_stable_loss": float(train_stable_loss),
                    "test_stable_loss": float(test_stable_loss),
                    "dataset": data_id,
                    "diffusion_step": diffusion_step,
                    "unconditional": unconditional
                }, f, indent=4)
            print("稳定分量损失评估完成，结果已保存\n")

        for temp_i in range(0,1):
            window_trick_evaluate_middle(model, train_error_loader_list, test_loader_list, nsample=1, scaler=1,
                              foldername=target_folder,
                              epoch_number=0, name=str(temp_i),split=split)

