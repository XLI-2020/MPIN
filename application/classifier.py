"""
The simple RNN classification model for imputed dataset PhysioNet-2012.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import argparse
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Global_Config import RANDOM_SEED
from modeling.utils import cal_classification_metrics
from modeling.utils import setup_logger

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class LoadImputedDataAndLabel(Dataset):
    def __init__(self, imputed_data, labels):
        self.imputed_data = imputed_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.imputed_data[idx].astype("float32")),
            torch.from_numpy(self.labels[idx].astype("float32")),
        )


class ImputedDataLoader:
    def __init__(
        self,
        original_data_path,
        imputed_data_path,
        seq_len,
        feature_num,
        batch_size=128,
        num_workers=4,
    ):
        """
        original_data_path: path of original dataset, which contains classification labels
        imputed_data_path: path of imputed data
        """
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers

        # with h5py.File(imputed_data_path, "r") as hf:
        #     imputed_train_set = hf["imputed_train_set"][:]
        #     imputed_val_set = hf["imputed_val_set"][:]
        #     imputed_test_set = hf["imputed_test_set"][:]

        # with h5py.File(original_data_path, "r") as hf:
        #     train_set_ori_X = hf["train"]["X"][:]
        #     imputed_train_set = np.nan_to_num(train_set_ori_X, nan=0)
        #     val_set_ori_X = hf["val"]["X"][:]
        #     imputed_val_set = np.nan_to_num(val_set_ori_X, nan=0)
        #     test_set_ori_X = hf["test"]["X"][:]
        #     imputed_test_set = np.nan_to_num(test_set_ori_X, nan=0)
        # eval_ratio = 0.2
        imputed_X_path = f'/home/xiaol/Documents/OCW/iif_exp/APP_ICU_{args.method}_testMissRate_window_4_eval_{args.eval_ratio}_stream_1.0_impute_X.txt'

        X = np.loadtxt(imputed_X_path)
        Y = np.loadtxt(imputed_X_path.replace('X', 'Y'))
        shp = X.shape
        logger.info(f'data shp: {shp}')
        num_of_samples, num_of_channels = X.shape
        X = X.reshape(48, -1, num_of_channels)
        X = np.transpose(X, (1, 0, 2))

        # print('X shp:', X.shape)
        # print('Y shp:', Y.shape)

        zeros_labels = np.where(Y == 0)[0]
        len_zeros = len(zeros_labels)
        logger.info('length of zeros:{len_zeros}')
        ones_labels = np.where(Y == 1)[0]
        # print('length of ones', len(ones_labels))

        """
          train data len: (7672, 1)
          val data len: (1918, 1)
          test data len: (2398, 1)
        """
        imputed_train_set = X[:7672]
        imputed_val_set = X[7672:9590]
        imputed_test_set = X[9590:]

        y_train = Y[:7672]
        y_val = Y[7672:9590]
        y_test = Y[9590:]
        with h5py.File(original_data_path, "r") as hf:
            train_set_labels = hf["train"]["labels"][:]
            val_set_labels = hf["val"]["labels"][:]
            test_set_labels = hf["test"]["labels"][:]

        # logger.info('inspect Y:', np.where(y_train==1)[0], np.where(train_set_labels==1)[0])
        # print('inspect Y:', y_train[:50], train_set_labels[:50])

        self.train_set = LoadImputedDataAndLabel(imputed_train_set, train_set_labels)
        self.val_set = LoadImputedDataAndLabel(imputed_val_set, val_set_labels)
        self.test_set = LoadImputedDataAndLabel(imputed_test_set, test_set_labels)

    def get_loaders(self):
        train_loader = DataLoader(
            self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            self.val_set, self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = DataLoader(self.test_set, self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


class SimpleRNNClassification(torch.nn.Module):
    def __init__(self, feature_num, rnn_hidden_size, class_num):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            feature_num, hidden_size=rnn_hidden_size, batch_first=True #(batch, seq, feature)
        )
        self.fcn = torch.nn.Linear(rnn_hidden_size, class_num)

    def forward(self, data):
        hidden_states, _ = self.rnn(data)
        logits = self.fcn(hidden_states[:, -1, :])
        prediction_probabilities = torch.sigmoid(logits)
        return prediction_probabilities


def train(model, train_dataloader, val_dataloader, optimizer):
    patience = 20
    current_patience = patience
    best_ROCAUC = 0
    for epoch in range(args.epochs):
        model.train()
        for idx, data in enumerate(train_dataloader):
            X, y = map(lambda x: x.to(args.device), data)
            optimizer.zero_grad()
            probabilities = model(X)
            loss = F.binary_cross_entropy(probabilities, y)
            loss.backward()
            optimizer.step()
            logger.info(f'train {epoch}: {idx}')

        logger.info('start val below')
        model.eval()
        probability_collector, label_collector = [], []
        with torch.no_grad():
            for idx, data in enumerate(val_dataloader):
                X, y = map(lambda x: x.to(args.device), data)
                probabilities = model(X)
                probability_collector += probabilities.cpu().tolist()
                label_collector += y.cpu().tolist()
                logger.info(f'val: {idx}')
        probability_collector = np.asarray(probability_collector)
        label_collector = np.asarray(label_collector)
        classification_metrics = cal_classification_metrics(
            probability_collector, label_collector
        )
        logger.info('classfication metric results:')
        logger.info(classification_metrics)
        if best_ROCAUC < classification_metrics["ROC_AUC"]:
            current_patience = patience
            best_ROCAUC = classification_metrics["ROC_AUC"]
            # save model
            saving_path = os.path.join(
                args.sub_model_saving,
                "model_epoch_{}_ROCAUC_{:.4f}".format(epoch, best_ROCAUC),
            )
            torch.save(model.state_dict(), saving_path)
        else:
            current_patience -= 1
        if current_patience == 0:
            break
    logger.info("All done. Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='./root/', help="model and log saving dir")
    parser.add_argument(
        "--original_dataset_path", type=str, default='/home/xiaol/Documents/SAITS/generated_datasets/physio2012_37feats_01masked/datasets.h5', help="path of original dataset"
    )
    parser.add_argument(
        "--imputed_dataset_path", type=str, default='/home/xiaol/Documents/SAITS/NIPS_results/PhysioNet2012_SAITS_best/step_313/imputations.h5', help="path of imputed dataset"
    )
    parser.add_argument("--seq_len", type=int, default=48, help="sequence length")
    parser.add_argument("--feature_num", type=int, default=37, help="feature num")
    parser.add_argument("--rnn_hidden_size", type=int, default=64, help="RNN hidden size")
    parser.add_argument("--epochs", type=int, default=100, help="max training epochs")
    parser.add_argument("--lr", type=float, default=0.01,  help="learning rate")
    parser.add_argument(
        "--test_mode",
        dest="test_mode",
        action="store_true",
        default=False,
        help="test mode to test saved model",
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default=None,
        help="test mode to test saved model",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run model"
    )

    parser.add_argument("--eval_ratio", type=float, default=0.1,  help="missing ratio")
    parser.add_argument("--method", type=str, default='MPIN',  help="imputation method")


    args = parser.parse_args()
    if args.test_mode:
        assert (
            args.saved_model_path is not None
        ), "saved_model_path must be provided in test mode"

    # create dirs
    time_now = datetime.now().__format__("%Y-%m-%d_T%H:%M:%S")
    log_saving = os.path.join(args.root_dir, f"logs_{args.method}")
    model_saving = os.path.join(args.root_dir, f"models_{args.method}")
    args.sub_model_saving = os.path.join(model_saving, f"{args.eval_ratio}_{time_now}")
    [
        os.makedirs(dir_)
        for dir_ in [model_saving, log_saving, args.sub_model_saving]
        if not os.path.exists(dir_)
    ]
    # create logger
    logger = setup_logger(os.path.join(log_saving, "log_" + time_now), "w")
    logger.info(f"args: {args}")

    # build models and dataloaders
    model = SimpleRNNClassification(args.feature_num, args.rnn_hidden_size, 1)
    dataloader = ImputedDataLoader(
        args.original_dataset_path,
        args.imputed_dataset_path,
        args.seq_len,
        args.feature_num,
        128,
    )
    train_set_loader, val_set_loader, test_set_loader = dataloader.get_loaders()
    if "cuda" in args.device and torch.cuda.is_available():
        model = model.to(args.device)

    if not args.test_mode:
        logger.info("Start training...")
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        train(model, train_set_loader, val_set_loader, optimizer)
    else:
        logger.info("Start testing...")
        checkpoint = torch.load(args.saved_model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        probability_collector, label_collector = [], []
        for idx, data in enumerate(test_set_loader):
            X, y = map(lambda x: x.to(args.device), data)
            probabilities = model(X)
            probability_collector += probabilities.cpu().tolist()
            label_collector += y.cpu().tolist()
        probability_collector = np.asarray(probability_collector)
        label_collector = np.asarray(label_collector)
        classification_metrics = cal_classification_metrics(
            probability_collector, label_collector
        )
        for k, v in classification_metrics.items():
            logger.info(f"{k}: {v}")



