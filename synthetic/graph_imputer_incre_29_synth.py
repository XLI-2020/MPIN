import sys
import pandas as pd
sys.path.append('/home/xiaol/Documents')
import torch
from OCW.models.regressor import MLPNet
import torch.optim as optim
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import copy
import numpy as np
import random
from datetime import datetime
from OCW.models.DynamicGNN import DynamicGCN, DynamicGAT, DynamicGraphSAGE
from argparse import ArgumentParser
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from load_dataset_synth import load_ICU_dataset, load_airquality_dataset, load_WiFi_dataset, get_model_size, load_synth_dataset
from torch_geometric.transforms import FeaturePropagation
from pypots.utils.metrics import cal_mae, cal_mse, cal_mre
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from fancyimpute import MatrixFactorization, SoftImpute, BiScaler, IterativeSVD
from sklearn.impute import SimpleImputer, KNNImputer

parser = ArgumentParser()
parser.add_argument("--incre_mode", type=str, default='alone')
parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--eval_ratio", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=100)

parser.add_argument("--window", type=int, default=13)

parser.add_argument('--base', type=str, default='SAGE')

parser.add_argument('--method', type=str, default='MICE')

parser.add_argument('--stream', type=float, default=1)

parser.add_argument("--prefix", type=str, default='testMissRate')

parser.add_argument("--dataset", type=str, default='LHS')

parser.add_argument("--num_of_iter", type=int, default=5)
parser.add_argument("--out_channels", type=int, default=256)
parser.add_argument("--k", type=int, default=10)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.1)

args = parser.parse_args()

st = datetime.now()
print('starting time:', st)

torch.random.manual_seed(2021)
device = torch.device('cuda')
out_channels = args.out_channels
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epochs

def data_transform(X, X_mask, eval_ratio=0.1):

    eval_mask = np.zeros(X_mask.shape)
    rows, cols = np.where(X_mask==1)
    print('rows, cols:', len(rows), len(cols))
    eval_row_index_index = random.sample(range(len(rows)),int(eval_ratio*len(rows)))
    eval_row_index = rows[eval_row_index_index]
    eval_col_index = cols[eval_row_index_index]
    # print('eval_row_index', eval_row_index)
    # print('eval_col_index', eval_col_index)
    X_mask[eval_row_index, eval_col_index] = 0
    eval_mask[eval_row_index, eval_col_index] = 1

    eval_X = copy.deepcopy(X)
    X[eval_row_index, eval_col_index] = 0
    return X, X_mask, eval_X, eval_mask


random.seed(2021)
if args.dataset in ["KDM", "WDS", "LHS"]:
    base_X = load_WiFi_dataset(window=args.window, dataset_name=args.dataset, time_step=5, method='mpin')
    base_X_mask = (~np.isnan(base_X)).astype(int)
    base_X = np.nan_to_num(base_X, nan=0)
    mean_X = np.mean(base_X)
    print('mean_X', mean_X)
    std_X = np.std(base_X)
    print('std_X', std_X)
    base_X = (base_X - mean_X) / std_X
    print('base WiFi {datasetname} data shape:'.format(datasetname=args.dataset), base_X.shape, base_X_mask.shape)

elif args.dataset == 'ICU':
    print('dataset:physionet')
    base_X = load_ICU_dataset(window=args.window, method='mpin', stream=args.stream)
    base_X_mask = (~np.isnan(base_X)).astype(int)
    base_X = np.nan_to_num(base_X)
    mean_X = np.mean(base_X)
    print('mean_X', mean_X)
    std_X = np.std(base_X)
    print('std_X', std_X)
    base_X = (base_X - mean_X) / std_X
    print('base physionet data shape:', base_X.shape, base_X_mask.shape)

elif args.dataset == 'Airquality':
    print('dataset:Airquality')
    base_X = load_airquality_dataset(window=args.window, method='mpin', stream=args.stream)
    base_X_mask = (~np.isnan(base_X)).astype(int)
    base_X = np.nan_to_num(base_X)
    mean_X = np.mean(base_X)
    print('mean_X', mean_X)
    std_X = np.std(base_X)
    print('std_X', std_X)
    base_X = (base_X - mean_X) / std_X
    print('base Airquality data shape:', base_X.shape, base_X_mask.shape)

elif args.dataset in ['1K_normal', '1M_normal']:
    base_X = load_synth_dataset(window=args.window, dataset_name=args.dataset, time_step=5, method='mpin')
    base_X_mask = (~np.isnan(base_X)).astype(int)
    # base_X = StandardScaler().fit_transform(base_X)
    base_X = np.nan_to_num(base_X, nan=0)
    print('base synthetic {datasetname} data shape:'.format(datasetname=args.dataset), base_X.shape, base_X_mask.shape)


elif args.dataset in ['1K', '1M']:
    base_X = load_synth_dataset(window=args.window, dataset_name=args.dataset, time_step=5, method=args.method)
    base_X_mask = (~np.isnan(base_X)).astype(int)
    # base_X = StandardScaler().fit_transform(base_X)
    base_X = np.nan_to_num(base_X, nan=0)
    mean_X = np.mean(base_X)
    print('mean_X', mean_X)
    std_X = np.std(base_X)
    print('std_X', std_X)
    base_X = (base_X - mean_X) / std_X
    print('base synthetic {datasetname} data shape:'.format(datasetname=args.dataset), base_X.shape, base_X_mask.shape)



def build_GNN(in_channels, out_channels, k, base):
    if base == 'GAT':
        gnn = DynamicGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'GCN':
        gnn = DynamicGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'SAGE':
        gnn = DynamicGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)

    return gnn

# def get_window_data(start, end, ratio):
#     slice_X = base_X[int(len(base_X)*start*ratio):int(len(base_X)*end*ratio)]
#     slice_X_mask = base_X_mask[int(len(base_X)*start*ratio):int(len(base_X)*end*ratio)]
#     return slice_X, slice_X_mask

def get_window_data(start, end, ratio):
    A = copy.copy(base_X)
    B = copy.copy(base_X_mask)
    X = A[int(len(A)*start*ratio):int(len(A)*end*ratio)]
    X_mask = B[int(len(A)*start*ratio):int(len(A)*end*ratio)]
    return X, X_mask

def window_imputation(start, end, sample_ratio, initial_state_dict=None, X_last=None, mask_last=None, transfer=False):

    X, X_mask = get_window_data(start=start, end=end, ratio=sample_ratio)
    ori_X = copy.copy(X)
    ori_X_row = ori_X.shape[0]

    ori_X_mask = copy.copy(X_mask)
    print('window_X:', ori_X.shape, ori_X_mask.shape)

    if X_last:
        X_last = np.array(X_last)
        # eval_X = np.concatenate([X_last, X], axis=0)

        X = np.concatenate([X_last, X], axis=0)

        # eval_mask_last = np.zeros(shp_last)
        # eval_mask = np.concatenate([eval_mask_last, eval_mask],axis=0)

        X_mask = np.concatenate([mask_last, X_mask], axis=0)

    X, X_mask, eval_X, eval_mask = data_transform(X, X_mask, eval_ratio=args.eval_ratio)

    print('initial X nan:', np.isnan(X).sum())
    print('initial eval_X nan:', np.isnan(eval_X).sum())


    # X, X_mask, eval_X, eval_mask = data_transform(X, X_mask, eval_ratio=eval_ratio)

    # if X_last:
    #     X_last = np.array(X_last)
    #     shp_last = X_last.shape
    #     eval_X = np.concatenate([X_last, X], axis=0)
    #
    #     X = np.concatenate([X_last, X], axis=0)
    #
    #     eval_mask_last = np.zeros(shp_last)
    #     eval_mask = np.concatenate([eval_mask_last, eval_mask],axis=0)
    #
    #     X_mask = np.concatenate([mask_last, X_mask], axis=0)

    in_channels = X.shape[1]
    print('in_channels:', in_channels)

    # out_channels = copy.copy(in_channels)
    eval_impute_error_list = []
    eval_impute_mse_error_list = []
    eval_impute_mape_error_list = []
    X_imputed_list = []
    state_dict_list = []

    elapsed_time_list = []
    st = datetime.now()

    if args.method == 'KNN':
        X_input = copy.deepcopy(X)
        X_input[np.where(eval_mask == 1)] = np.nan
        # X_input[np.where(X_mask == 0)] = np.nan

        imputer = KNNImputer(n_neighbors=args.k)
        imputer = imputer.fit(X_input)
        X_imputed = imputer.transform(X_input)

    elif args.method == 'MICE':
        print('MICE!!!')
        X_input = copy.deepcopy(X)
        X_input[np.where(eval_mask == 1)] = np.nan
        # X_input[np.where(X_mask == 0)] = np.nan

        imputer = IterativeImputer(random_state=12)

        if args.dataset in ['KDM', 'WDS', 'LHS']:
            X_cols = X_input.shape[1]
            batch_size = 40
            nb_ft_bs = (X_cols + batch_size - 1) // batch_size
            x_mice_col_list = []
            for j in range(nb_ft_bs):
                print('j:,{j}'.format(j=j))
                x = X_input[:, j * batch_size: (j + 1) * batch_size]
                print('x shaope', x.shape)
                x_mice_col = imputer.fit_transform(x)
                print('x_mice_col shape', x_mice_col.shape)
                x_mice_col_list.append(x_mice_col)
            X_imputed = np.concatenate(x_mice_col_list, axis=1)
        else:
            imputer = imputer.fit(X_input)
            X_imputed = imputer.transform(X_input)

    elif args.method.startswith('MF'):
        concrete_meth = args.method.split('-')[-1]
        X_input = copy.deepcopy(X)
        X_input[np.where(eval_mask == 1)] = np.nan
        # X_input[np.where(X_mask == 0)] = np.nan

        if concrete_meth == 'soft':
            MF_imputer = SoftImpute()
        elif concrete_meth == 'bi':
            MF_imputer = BiScaler()
        elif concrete_meth == 'iter':
            MF_imputer = IterativeSVD()
        elif concrete_meth == 'mf':
            MF_imputer = MatrixFactorization()
        X_imputed = MF_imputer.fit_transform(X_input)

    elif args.method in ['mean', 'median', 'most_frequent', 'constant']:
        X_input = copy.deepcopy(X)
        X_input[np.where(eval_mask == 1)] = np.nan

        # X_input[np.where(X_mask == 0)] = np.nan
        imputer = SimpleImputer(missing_values=np.nan, strategy=args.method)
        imputer = imputer.fit(X_input)
        X_imputed = imputer.transform(X_input)

    trans_X = copy.copy(X_imputed)

    # trans_eval_X = eval_X * std_f + mean_f
    trans_eval_X = copy.copy(eval_X)

    print('trans_X shape', trans_X.shape)
    print('trans_eval_X shape', trans_eval_X.shape)


    print('nan of X_imputed:', np.isnan(X_imputed).sum())

    print('nan of eval_X:', np.isnan(eval_X).sum())


    eval_impute_error = cal_mae(trans_X, trans_eval_X, eval_mask)

    eval_impute_error_mse = cal_mse(trans_X, trans_eval_X, eval_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

    eval_impute_error_mape = cal_mre(trans_X, trans_eval_X, eval_mask)


    print('valid impute value samples:', (trans_X[np.where(eval_mask == 1)]))
    print('valid true value samples:', (trans_eval_X[np.where(eval_mask == 1)]))

    print('valid impute error MAE:', eval_impute_error)
    print('valid impute error MSE:', eval_impute_error_mse)

    print('valid impute error MRE:', eval_impute_error_mape)

    eval_impute_error_list.append(round(eval_impute_error, 6))
    print('valid min impute error MAE:', min(eval_impute_error_list))

    eval_impute_mse_error_list.append(round(eval_impute_error_mse, 6))
    print('valid min impute error MSE:', min(eval_impute_mse_error_list))

    eval_impute_mape_error_list.append(round(eval_impute_error_mape, 6))
    print('valid min impute error MRE:', min(eval_impute_mape_error_list))


    current_time = datetime.now()
    elapsed_time = (current_time - st).total_seconds() / 60
    print('elapsed time:', elapsed_time)
    elapsed_time_list.append(round(elapsed_time, 6))

    arg_min_mae_error, min_mae_error = min(list(enumerate(eval_impute_error_list)), key=lambda x: x[1])

    print('min MAE error: epoch, MAE, time, MSE', arg_min_mae_error, min_mae_error,
          elapsed_time_list[arg_min_mae_error],
          eval_impute_mse_error_list[arg_min_mae_error])

    arg_min_mse_error, min_mse_error = min(list(enumerate(eval_impute_mse_error_list)), key=lambda x: x[1])

    print('min MSE error: epoch, MSE, time, MAE', arg_min_mse_error, min_mse_error,
          elapsed_time_list[arg_min_mse_error],
          eval_impute_error_list[arg_min_mse_error])

    et = datetime.now()
    total_elapsed_time = round((et-st).total_seconds()/60, 6)
    results_list = [arg_min_mae_error, min_mae_error, eval_impute_mse_error_list[arg_min_mae_error], eval_impute_mape_error_list[arg_min_mae_error], elapsed_time_list[arg_min_mae_error], total_elapsed_time]
    return ori_X_mask, results_list

incre_mode = args.incre_mode # 'alone',  'data', 'state', 'state+transfer', 'data+state', 'data+state+transfer'
# sample_ratio = args.sample_ratio
eval_ratio = args.eval_ratio
prefix = args.prefix
num_windows = 1
results_schema = ['opt_epoch', 'opt_mae', 'mse', 'mape', 'opt_time', 'tot_time']

num_of_iteration = args.num_of_iter
iter_results_list = []

for iteration in range(num_of_iteration):
    results_collect = []
    for w in range(num_windows):
        print(f'which time window:{w}')
        if w == 0:
            mask_last, window_results = window_imputation(start=w, end=w+1, sample_ratio=1)
            results_collect.append(window_results)
        else:
            continue
            # if incre_mode == 'alone':
            #     X_last, mask_last, window_results = window_imputation(start=w, end=w+1, sample_ratio=sample_ratio)
            #     results_collect.append(window_results)
            #
            # elif incre_mode == 'data':
            #     X_last, mask_last, window_results = window_imputation(start=w, end=w + 1, sample_ratio=sample_ratio,
            #                                                                                  X_last=X_last, mask_last=mask_last)
            #     results_collect.append(window_results)

            # elif incre_mode == 'state':
            #     X_last, mask_last, window_results = window_imputation(start=w, end=w + 1, sample_ratio=sample_ratio,initial_state_dict=window_best_state)
            #     results_collect.append(window_results)
            #
            # elif incre_mode == 'state+transfer':
            #     X_last, mask_last, window_results = window_imputation(start=w, end=w + 1, sample_ratio=sample_ratio,initial_state_dict=window_best_state, transfer=
            #                                                                                  True)
            #     results_collect.append(window_results)

            # elif incre_mode == 'data+state':
            #     X_last, mask_last, window_results = window_imputation(start=w, end=w+1, sample_ratio=sample_ratio, initial_state_dict=window_best_state, X_last=X_last, mask_last=mask_last)
            #     results_collect.append(window_results)
            # elif incre_mode == 'data+state+transfer':
            #     X_last, mask_last, window_results = window_imputation(start=w, end=w+1, sample_ratio=sample_ratio, initial_state_dict=window_best_state, X_last=X_last, mask_last=mask_last, transfer=True)
            #     results_collect.append(window_results)

    df = pd.DataFrame(results_collect, index=range(num_windows), columns=results_schema)
    iter_results_list.append(df)

print('ready to write data!')
avg_df = sum(iter_results_list)/num_of_iteration
avg_df = avg_df.round(4)
if args.prefix == 'testNumStream':
    avg_df.to_csv(f'./exp_results/synth_{args.method}_{args.prefix}_{args.dataset}_{args.k}_incre_{args.incre_mode}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}.csv', header=True, index=False)
else:
    avg_df.to_csv(f'./exp_results/synth_{args.method}_{args.prefix}_{args.dataset}_{args.k}_incre_{args.incre_mode}_window_{args.window}_eval_{args.eval_ratio}.csv', header=True, index=False)
print('done!')
print('finishing time:', datetime.now())































