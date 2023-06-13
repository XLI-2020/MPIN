# Install PyPOTS first: pip install pypots
import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS, BRITS
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from pypots.utils.metrics import cal_mae, cal_mse, cal_mre
import sys
sys.path.append('/home/xiaol/Documents')

from OCW.load_dataset import load_ICU_dataset, load_airquality_dataset, load_WiFi_dataset, get_model_size
from datetime import datetime


st = datetime.now()
np.random.seed(13)

parser = ArgumentParser()
parser.add_argument("--eval_ratio", type=float, default=0.1)
parser.add_argument("--window", type=int, default=2)
parser.add_argument("--dataset", type=str, default='Airquality')
parser.add_argument("--prefix", type=str, default='testMissRate')
parser.add_argument('--stream', type=float, default=1)


parser.add_argument("--method", type=str, default='saits')


args = parser.parse_args()
# Data preprocessing. Tedious, but PyPOTS can help. 🤓

device = torch.device('cuda')


if args.dataset == 'ICU':
    X = load_ICU_dataset(window=args.window, stream=args.stream)
elif args.dataset == 'Airquality':
    X = load_airquality_dataset(window=args.window, stream=args.stream)
elif args.dataset in ["KDM", "WDS", "LHS"]:
    X = load_WiFi_dataset(window=args.window, dataset_name=args.dataset, time_step=5)

num_of_samples, num_of_ts, num_of_channel = X.shape

X_intact, X, missing_mask, indicating_mask = mcar(X, args.eval_ratio) # hold out 10% observed values as ground truth
X = masked_fill(X, 1 - missing_mask, np.nan)
# Model training. This is PyPOTS showtime. 💪

if args.method == 'brits':
    imputer = BRITS(n_steps=num_of_ts, n_features=num_of_channel, rnn_hidden_size=64, epochs=1000, device=device)
elif args.method == 'saits':
    imputer = SAITS(n_steps=num_of_ts, n_features=num_of_channel, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=1000, device=device)
imputer.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
imputation = imputer.impute(X)  # impute the originally-missing values and artificially-missing values

mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
mse = cal_mse(imputation, X_intact, indicating_mask)
mre = cal_mre(imputation, X_intact, indicating_mask)

num_params = sum(p.numel() for p in imputer.model.parameters() if p.requires_grad)/1e6
print('num of Parameters:', num_params)

model_size = get_model_size(imputer.model)
print('SAITS Model Size:', model_size)

print('type of imputations 1', type(imputation))
print('X_intact shape', X_intact.shape, type(X_intact))
print('X_indicate mask shape', indicating_mask.shape)
print('imputation shape 1', imputation.shape)

torch.random.manual_seed(5)
# device = torch.device('cuda')
imputation = torch.FloatTensor(imputation)
print('imputation shape 2', imputation.shape)
imputation = torch.FloatTensor(imputation.reshape(-1, num_of_channel))
X_intact = torch.FloatTensor(X_intact.reshape(-1, num_of_channel))
indicating_mask = torch.LongTensor(indicating_mask.reshape(-1, num_of_channel))

trans_X = copy.deepcopy(imputation)
trans_eval_X = copy.deepcopy(X_intact)
eval_mask = copy.deepcopy(indicating_mask)

eval_impute_error = copy.deepcopy(mae)

eval_impute_error_mse = copy.deepcopy(mse)

eval_impute_error_mre = copy.deepcopy(mre)


# eval_impute_error = torch.sum(torch.abs(trans_X - trans_eval_X) * eval_mask) / (torch.sum(eval_mask) + 1e-5)
# eval_impute_error_mse = F.mse_loss(trans_X[torch.where(eval_mask == 1)], trans_eval_X[torch.where(eval_mask == 1)])

print('valid impute value samples:', (trans_X[torch.where(eval_mask == 1)]))
print('valid true value samples:', (trans_eval_X[torch.where(eval_mask == 1)]))

print('valid impute error MAE:', eval_impute_error.item())
print('valid impute error MSE:', eval_impute_error_mse.item())
print('valid impute error MRE:', eval_impute_error_mre.item())


et = datetime.now()

elapsed_time = (et - st).total_seconds()/60
print('end time:', et)
print('elasped time:(mins)', elapsed_time)
print('done!!!')

results = [[eval_impute_error.item(), eval_impute_error_mse.item(), eval_impute_error_mre.item(), num_params, model_size, elapsed_time]]

results_schema = ['opt_mae', 'mse', 'mape', 'para', 'memo', 'opt_time']

res_df = pd.DataFrame(results, columns=results_schema)
res_df = res_df.round(4)

if args.prefix == 'testNumStream':
    res_df.to_csv(f'./exp_res/{args.method}_{args.prefix}_{args.dataset}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}.csv', header=True, index=False)
else:
    res_df.to_csv(f'./exp_res/{args.method}_{args.prefix}_{args.dataset}_window_{args.window}_eval_{args.eval_ratio}.csv', header=True, index=False)

