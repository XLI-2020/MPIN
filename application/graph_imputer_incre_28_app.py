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
from load_dataset_app import load_ICU_dataset, load_airquality_dataset, load_WiFi_dataset, load_ICU_dataset_all, load_airquality_dataset_all, load_ICU_dataset_all_app
from torch_geometric.transforms import FeaturePropagation
from pypots.utils.metrics import cal_mae, cal_mse, cal_mre


parser = ArgumentParser()
parser.add_argument("--incre_mode", type=str, default='alone')
parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--eval_ratio", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--window", type=int, default=2)
parser.add_argument('--stream', type=float, default=1)

parser.add_argument('--base', type=str, default='GAT')

parser.add_argument('--method', type=str, default='FP')

parser.add_argument("--prefix", type=str, default='testMissRate')

parser.add_argument("--dataset", type=str, default='ICU')

parser.add_argument("--num_of_iter", type=int, default=5)
parser.add_argument("--out_channels", type=int, default=256)
parser.add_argument("--k", type=int, default=20)

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
    eval_ratio = eval_ratio
    eval_row_index_index = random.sample(range(len(rows)),int(eval_ratio*len(rows)))
    eval_row_index = rows[eval_row_index_index]
    eval_col_index = cols[eval_row_index_index]
    # print('eval_row_index', eval_row_index)
    # print('eval_col_index', eval_col_index)
    X_mask[eval_row_index, eval_col_index] = 0
    eval_mask[eval_row_index, eval_col_index] = 1

    eval_X = copy.copy(X)
    X[eval_row_index, eval_col_index] = 0
    return X, X_mask, eval_X, eval_mask


random.seed(2021)
if args.dataset in ["KDM", "WDS", "LHS"]:
    base_X, base_Y = load_WiFi_dataset(window=args.window, dataset_name=args.dataset, time_step=5, method='mpin')
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
    base_X, base_Y, sc_scaler = load_ICU_dataset_all_app(method='mpin', stream=args.stream)
    base_X_mask = (~np.isnan(base_X)).astype(int)
    base_X = np.nan_to_num(base_X)
    # mean_X = np.mean(base_X)
    # print('mean_X', mean_X)
    # std_X = np.std(base_X)
    # print('std_X', std_X)
    # base_X = (base_X - mean_X) / std_X
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


def build_GNN(in_channels, out_channels, k, base):
    if base == 'GAT':
        gnn = DynamicGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'GCN':
        gnn = DynamicGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'SAGE':
        gnn = DynamicGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)

    return gnn

# def get_window_data(start, end, ratio):
#     X = base_X[int(len(base_X)*start*ratio):int(len(base_X)*end*ratio)]
#     X_mask = base_X_mask[int(len(base_X)*start*ratio):int(len(base_X)*end*ratio)]
#     return X, X_mask

def get_window_data(start, end, ratio):
    A = copy.copy(base_X)
    B = copy.copy(base_X_mask)
    X = A[int(len(A)*start*ratio):int(len(A)*end*ratio)]
    X_mask = B[int(len(A)*start*ratio):int(len(A)*end*ratio)]
    return X, X_mask

def window_imputation(start, end, sample_ratio, initial_state_dict=None, X_last=None, mask_last=None, transfer=False):

    ori_X, ori_X_mask = get_window_data(start=start, end=end, ratio=sample_ratio)

    print('window_X:', ori_X.shape, ori_X_mask.shape)
    X = copy.copy(ori_X)
    ori_X_row = ori_X.shape[0]

    X_mask = copy.copy(ori_X_mask)

    if X_last:
        X_last = np.array(X_last)
        # eval_X = np.concatenate([X_last, X], axis=0)

        X = np.concatenate([X_last, X], axis=0)

        # eval_mask_last = np.zeros(shp_last)
        # eval_mask = np.concatenate([eval_mask_last, eval_mask],axis=0)

        X_mask = np.concatenate([mask_last, X_mask], axis=0)

    X, X_mask, eval_X, eval_mask = data_transform(X, X_mask, eval_ratio=eval_ratio)


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


    X = torch.FloatTensor(X).to(device)
    X_mask = torch.LongTensor(X_mask).to(device)
    eval_X = torch.FloatTensor(eval_X).to(device)
    eval_mask = torch.LongTensor(eval_mask).to(device)
    X_imputed = copy.copy(X)

    edge_index = knn_graph(X_imputed, args.k, batch=None, loop=False)
    data = Data(x=X_imputed, edge_index=edge_index, num_of_iteration=args.epochs)
    transform = FeaturePropagation(missing_mask=(~(X_mask.bool())))
    data = transform(data)
    X_imputed = data.x
    print(222, data)


    trans_X = copy.copy(X_imputed)
    # X_imputed_list.append(X_imputed.data.cpu().numpy().tolist())

    # trans_eval_X = eval_X * std_f + mean_f
    trans_eval_X = copy.copy(eval_X)

    print('trans_X shape', trans_X.shape)
    print('trans_eval_X shape', trans_eval_X.shape)

    # eval_impute_error = torch.sum(torch.abs(trans_X - trans_eval_X) * eval_mask)/torch.sum(eval_mask)
    # eval_impute_error_mse = F.mse_loss(trans_X[torch.where(eval_mask == 1)],
    #                                    trans_eval_X[torch.where(eval_mask == 1)])
    # eval_impute_error_mape = torch.sum(torch.abs((trans_X[torch.where(eval_mask == 1)]-
    #                                    trans_eval_X[torch.where(eval_mask == 1)])/trans_eval_X[torch.where(eval_mask == 1)]))/torch.sum(eval_mask)

    eval_impute_error = cal_mae(trans_X, trans_eval_X, eval_mask)

    eval_impute_error_mse = cal_mse(trans_X, trans_eval_X, eval_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

    eval_impute_error_mape = cal_mre(trans_X, trans_eval_X, eval_mask)

    best_X_imputed = X_imputed.data.cpu().numpy() * (1 - ori_X_mask) + ori_X * ori_X_mask
    # output_best_X_imputed = best_X_imputed * std_X + mean_X
    output_best_X_imputed = sc_scaler.inverse_transform(best_X_imputed)

    print('recover insepct:')
    print('ori X:', ori_X[:5])
    print('best_X_imputed', best_X_imputed[:3])
    print('output_best_X_imputed')


    print('valid impute value samples:', (trans_X[torch.where(eval_mask == 1)]))
    print('valid true value samples:', (trans_eval_X[torch.where(eval_mask == 1)]))

    print('valid impute error MAE:', eval_impute_error.item())
    print('valid impute error MSE:', eval_impute_error_mse.item())

    print('valid impute error MRE:', eval_impute_error_mape.item())

    eval_impute_error_list.append(round(eval_impute_error.item(), 6))
    print('valid min impute error MAE:', min(eval_impute_error_list))

    eval_impute_mse_error_list.append(round(eval_impute_error_mse.item(), 6))
    print('valid min impute error MSE:', min(eval_impute_mse_error_list))

    eval_impute_mape_error_list.append(round(eval_impute_error_mape.item(), 6))
    print('valid min impute error MRE:', min(eval_impute_mape_error_list))

    # eval_impute_mape_error_list.append(round(eval_impute_error_mape.item(), 6))
    #
    # print('valid impute value samples:', (trans_X[torch.where(eval_mask == 1)]))
    # print('valid true value samples:', (trans_eval_X[torch.where(eval_mask == 1)]))
    #
    # print('valid impute error MAE:', eval_impute_error.item())
    # print('valid impute error MSE:', eval_impute_error_mse.item())
    #
    # eval_impute_error_list.append(round(eval_impute_error.item(), 6))
    # print('valid min impute error MAE:', min(eval_impute_error_list))
    #
    # eval_impute_mse_error_list.append(round(eval_impute_error_mse.item(), 6))
    # print('valid min impute error MSE:', min(eval_impute_mse_error_list))

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
    return  results_list, output_best_X_imputed

incre_mode = args.incre_mode # 'alone',  'data', 'state', 'state+transfer', 'data+state', 'data+state+transfer'
# sample_ratio = args.sample_ratio
eval_ratio = args.eval_ratio
prefix = args.prefix


if args.dataset == 'KDM':
    num_windows = int(60/args.window)
elif args.dataset == 'ICU':
    num_windows = int(48/args.window)
elif args.dataset == 'Airquality':
    num_windows = int(24/args.window)
elif args.dataset == 'LHS':
    num_windows = int(200/args.window)

results_schema = ['opt_epoch', 'opt_mae', 'mse', 'mape', 'opt_time', 'tot_time']

num_of_iteration = args.num_of_iter
iter_results_list = []
iter_imputation_list = []

for iteration in range(num_of_iteration):
    results_collect = []
    imputation_collect = []
    for w in range(num_windows):
        print(f'which time window:{w}')
        if w == 0 :
            window_results, output_best_X_imputed = window_imputation(start=w, end=w+1, sample_ratio=1/float(num_windows))
            results_collect.append(window_results)
            imputation_collect.append(output_best_X_imputed)

        else:
            if incre_mode == 'alone':
                window_results, output_best_X_imputed = window_imputation(start=w, end=w+1, sample_ratio=1/float(num_windows))
                results_collect.append(window_results)
                imputation_collect.append(output_best_X_imputed)
            else:
                print('Abnormal!!!')

    df = pd.DataFrame(results_collect, index=range(num_windows), columns=results_schema)
    iter_results_list.append(df)

    output_all_best_X_imputed = np.concatenate(imputation_collect, axis=0)
    print('all output_all_best_X_imputed shp:', output_all_best_X_imputed.shape)
    iter_imputation_list.append(output_all_best_X_imputed)


print('ready to write data!')
avg_df = sum(iter_results_list)/num_of_iteration
avg_df = avg_df.round(4)

avg_X_imputed = sum(iter_imputation_list)/num_of_iteration
print('avg_X_imputed sample:', avg_X_imputed[:5])


if args.dataset == 'KDM':
    output_X_Y = np.concatenate([avg_X_imputed, base_Y], axis=1)
    np.savetxt(f'./iif_exp/APP_{args.dataset}_{args.method}_{args.prefix}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}_imputation_XY.txt', output_X_Y, fmt='%.2f')
elif args.dataset == 'ICU':
    np.savetxt(f'./iif_exp/APP_{args.dataset}_{args.method}_{args.prefix}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}_impute_X.txt', avg_X_imputed, fmt='%.2f')
    np.savetxt(f'./iif_exp/APP_{args.dataset}_{args.method}_{args.prefix}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}_impute_Y.txt', base_Y, fmt='%.2f')


avg_df.to_csv(f'./iif_exp/APP_{args.dataset}_{args.method}_{args.prefix}_{args.incre_mode}_{args.k}_window_{args.window}_epoch_{args.epochs}_eval_{args.eval_ratio}_stream_{args.stream}', header=True, index=True)


print('done!')
print('finishing time:', datetime.now())































