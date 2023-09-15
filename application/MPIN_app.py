import sys
import pandas as pd
sys.path.append('/home/xiaol/Documents')
sys.path.append('/home/xiao/Documents')

import torch
from OCW.models.regressor import MLPNet
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
import random
from datetime import datetime
from OCW.models.DynamicGNN import DynamicGCN, DynamicGAT, DynamicGraphSAGE, StaticGCN, StaticGraphSAGE, StaticGAT
from argparse import ArgumentParser
from torch_geometric.nn import knn_graph
from load_dataset_app import load_ICU_dataset_all, load_airquality_dataset_all, load_WiFi_dataset_all, get_model_size, load_ICU_dataset
from pypots.utils.metrics import cal_mae, cal_mse, cal_mre
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_dense_adj


# x = F.relu(self.gc1(x, adj))
# x = F.dropout(x, self.dropout, training=self.training)

parser = ArgumentParser()
parser.add_argument("--incre_mode", type=str, default='alone')

parser.add_argument("--window", type=int, default=6)
parser.add_argument('--stream', type=float, default=1)

parser.add_argument("--eval_ratio", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--site", type=str, default='KDM')
parser.add_argument("--floor", type=str, default='F1')

parser.add_argument('--base', type=str, default='SAGE')
parser.add_argument("--prefix", type=str, default='testMissRate')

parser.add_argument('--state', type=str, default='true')
parser.add_argument('--thre', type=float, default=0.25)

parser.add_argument('--method', type=str, default='DMU')

parser.add_argument("--num_of_iter", type=int, default=5)
parser.add_argument("--out_channels", type=int, default=256)
parser.add_argument("--k", type=int, default=10)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--dynamic", type=str, default='false')
parser.add_argument("--dataset", type=str, default='ICU')

# np.set_printoptions(suppress=True)


args = parser.parse_args()
starting_time = datetime.now()
print('starting time:', starting_time)

torch.random.manual_seed(2021)
device = torch.device('cuda')
out_channels = args.out_channels
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epochs

print('state', args.state)
print('thre', args.thre)
print('now', datetime.now())

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
    eval_X = copy.copy(X)
    X[eval_row_index, eval_col_index] = 0
    return X, X_mask, eval_X, eval_mask

random.seed(2021)
if args.dataset in ["KDM", "WDS", "LHS"]:
    base_X, base_Y = load_WiFi_dataset_all(dataset_name=args.dataset, time_step=5, method='mpin')
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
    # base_X = load_ICU_dataset_all(method='mpin', stream=args.stream)
    base_X, base_Y = load_ICU_dataset_all(window=args.window, method='mpin', stream=args.stream)

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
    base_X = load_airquality_dataset_all(method='mpin', stream=args.stream)
    base_X_mask = (~np.isnan(base_X)).astype(int)
    base_X = np.nan_to_num(base_X)
    mean_X = np.mean(base_X)
    print('mean_X', mean_X)
    std_X = np.std(base_X)
    print('std_X', std_X)
    base_X = (base_X - mean_X) / std_X
    print('base Airquality data shape:', base_X.shape, base_X_mask.shape)

# elif args.dataset == 'ICU':
#     print('dataset:physionet')
#     base_X = load_ICU_dataset(args.sample_ratio, method='mpin')
#     base_X_mask = (~np.isnan(base_X)).astype(int)
#     base_X_cp = np.nan_to_num(base_X)
#     mean_X = np.mean(base_X_cp)
#     print('mean_X', mean_X)
#     base_X = np.nan_to_num(base_X, nan=mean_X)
#     # base_X = StandardScaler().fit_transform(base_X)
#     mean_X = np.mean(base_X)
#     std_X = np.std(base_X)
#     print('std_X', std_X)
#     base_X = (base_X - mean_X) / std_X
#     print('base physionet data shape:', base_X.shape, base_X_mask.shape)



def build_GNN(in_channels, out_channels, k, base):
    if base == 'GAT':
        gnn = DynamicGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'GCN':
        gnn = DynamicGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'SAGE':
        gnn = DynamicGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)

    return gnn

def build_GNN_static(in_channels, out_channels, k, base):
    if base == 'GAT':
        gnn = StaticGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'GCN':
        gnn = StaticGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'SAGE':
        gnn = StaticGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    return gnn

#
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


def window_imputation(start, end, sample_ratio, initial_state_dict=None, X_last=None, mask_last=None, mae_last=None, transfer=False, state=args.state):

    ori_X, ori_X_mask = get_window_data(start=start, end=end, ratio=sample_ratio)
    X = copy.copy(ori_X)
    X_mask = copy.copy(ori_X_mask)
    feature_dim = X.shape[1]
    ori_X = copy.copy(X)
    ori_X_row = ori_X.shape[0]

    ori_X_mask = copy.deepcopy(X_mask)
    all_mask = copy.copy(X_mask)
    all_X = copy.copy(X)

    # if X_last:
    #     X_last = np.array(X_last)
    #     # eval_X = np.concatenate([X_last, X], axis=0)
    #     X = np.concatenate([X_last, X], axis=0)
    #     # eval_mask_last = np.zeros(shp_last)
    #     # eval_mask = np.concatenate([eval_mask_last, eval_mask],axis=0)
    #
    #     X_mask = np.concatenate([mask_last, X_mask], axis=0)
    #
    #     all_mask = copy.copy(X_mask)
    #     all_X = copy.copy(X)

    if X_last:
        X_last = np.array(X_last)
        # eval_X = np.concatenate([X_last, X], axis=0)
        all_X = np.concatenate([X_last, X], axis=0)
        # eval_mask_last = np.zeros(shp_last)
        # eval_mask = np.concatenate([eval_mask_last, eval_mask],axis=0)
        all_mask = np.concatenate([mask_last, X_mask], axis=0)

        X_last = X_last.tolist()

    print('all mask shp', all_mask.shape)
    print('all X shp', all_X.shape)

    all_mask_ts = torch.FloatTensor(all_mask).to(device)

    # gram_matrix = all_mask_ts.matmul(all_mask_ts.transpose(1,0))

    gram_matrix = torch.mm(all_mask_ts, all_mask_ts.t())  # compute the gram product

    # gram_matrix = all_mask @ (all_mask.transpose())
    print('gram_matrix shp', gram_matrix.shape)
    # print('gram_matrix', gram_matrix)

    gram_vec = gram_matrix.diagonal()
    print('gram vec shp', gram_vec.shape)
    print('gram vec', gram_vec)

    gram_row_sum = gram_matrix.sum(dim=0)

    print('gram_row_sum shp', gram_row_sum.shape)

    print('gram_row_sum', gram_row_sum)

    value_vec = gram_vec - (gram_row_sum - gram_vec)/(gram_matrix.shape[0]-1)

    print('value_vec shp', value_vec.shape)
    print('value_vec:', value_vec)

    # print('max min mean median vec values shp:', max(value_vec), min(value_vec), np.mean(value_vec), np.median(value_vec))

    keep_index = torch.where(value_vec > args.thre * (feature_dim-1))[0]
    keep_index = keep_index.data.cpu().numpy()
    # keep_index = torch.where(value_vec > np.quantile(value_vec, args.thre))

    keep_mask = all_mask[keep_index]

    keep_X = all_X[keep_index]

    print('keep_index', keep_index)
    print('keep_mask shp', keep_mask.shape)
    print('keep_X shp', keep_X.shape)

    # all_mask_sum = np.sum(all_mask, axis=1)

    # keep_index = np.where(all_mask_sum>np.quantile(all_mask_sum, args.quantile))
    # keep_mask = X_mask[keep_index]
    # keep_X = X[keep_index]

    # gram_mask = torch.spmm(all_mask, torch.transpose(all_mask, (1,0)))

    # X, X_mask, eval_X, eval_mask = data_transform(X, X_mask, eval_ratio=args.eval_ratio)

    X, X_mask, eval_X, eval_mask = data_transform(X, X_mask, eval_ratio=args.eval_ratio)

    if X_last:
        X_last = np.array(X_last)
        shp_last = X_last.shape
        eval_X = np.concatenate([X_last, eval_X], axis=0)
        X = np.concatenate([X_last, X], axis=0)
        eval_mask_last = np.zeros(shp_last)
        eval_mask = np.concatenate([eval_mask_last, eval_mask],axis=0)
        X_mask = np.concatenate([mask_last, X_mask], axis=0)

    in_channels = X.shape[1]
    print('in_channels:', in_channels)
    X = torch.FloatTensor(X).to(device)
    X_mask = torch.LongTensor(X_mask).to(device)
    eval_X = torch.FloatTensor(eval_X).to(device)
    eval_mask = torch.LongTensor(eval_mask).to(device)
    # mean_f = torch.FloatTensor(mean_f).to(device)
    # std_f = torch.FloatTensor(std_f).to(device)

    # build model
    if args.dynamic == 'true':
        print('dynamic true:', args.dynamic)
        gnn = build_GNN(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)
        gnn2 = build_GNN(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)
    else:
        print('dynamic false:', args.dynamic)
        gnn = build_GNN_static(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)
        gnn2 = build_GNN_static(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)
        # gnn3 = build_GNN_static(in_channels=in_channels, out_channels=out_channels, k=args.k, base=args.base)

    model_list = [gnn, gnn2]
    regressor = MLPNet(out_channels, in_channels).to(device)

    if initial_state_dict != None:
        gnn.load_state_dict(initial_state_dict['gnn'])
        gnn2.load_state_dict(initial_state_dict['gnn2'])
        if not transfer:
            regressor.load_state_dict(initial_state_dict['regressor'])

    trainable_parameters = []
    for model in model_list:
        trainable_parameters.extend(list(model.parameters()))

    trainable_parameters.extend(list(regressor.parameters()))
    filter_fn = list(filter(lambda p: p.requires_grad, trainable_parameters))

    # num_params_gnn = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    # num_params_gnn2 = sum(p.numel() for p in gnn2.parameters() if p.requires_grad)
    # num_params_regressor = sum(p.numel() for p in regressor.parameters() if p.requires_grad)

    num_of_params = sum(p.numel() for p in filter_fn)

    print('number of trainable parameters:', num_of_params)
    model_size = get_model_size(gnn) + get_model_size(gnn2) + get_model_size(regressor)
    model_size = round(model_size, 6)

    print('model size:', model_size)

    num_of_params = num_of_params/1e6

    opt = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)

    graph_impute_layers = len(model_list)

    st = datetime.now()

    # X_knn = X * X_mask
    X_knn = copy.deepcopy(X)

    edge_index = knn_graph(X_knn, args.k, batch=None, loop=False, cosine=False)


    # print('X X_mask shp:', X.shape, X_mask.shape)
    #
    # adj = to_dense_adj(edge_index).data.cpu().numpy().squeeze().transpose()
    #
    # print('adj shp:', adj.shape)
    # print(adj)
    #
    # adj_sum = np.sum(adj, axis=1).flatten()
    #
    # print('adj sum shp:', adj_sum.shape)
    # print(adj_sum)
    #
    # degree_matrix = np.diag(1/adj_sum).round(2)
    # print('degree shp:', degree_matrix.shape)
    # print(degree_matrix)
    #
    # norm_adj = (degree_matrix@adj).round(2)
    # print('norm_adj shp:', norm_adj.shape)
    # print(norm_adj)
    #
    # gram_mask = all_mask @ (all_mask.transpose())
    #
    # print('gram_mask shp:', gram_mask.shape)
    # print(gram_mask)
    #
    # value_matrix = gram_mask - gram_mask@(norm_adj.T)
    #
    # print('value_matrix shp:', value_matrix.shape)
    # print(value_matrix)
    #
    # value_vec = value_matrix.diagonal()
    #
    # print('value_vec shp:', value_vec)
    #
    # print('max min mean median vec values shp:', max(value_vec), min(value_vec), np.mean(value_vec), np.median(value_vec))
    #
    # keep_index = np.where(value_vec > np.quantile(value_vec, args.quantile))
    # keep_mask = all_mask[keep_index]
    # keep_X = all_X[keep_index]


    min_mae_error = 1e9
    min_mse_error = None
    min_mape_error = None
    opt_epoch = None
    opt_time = None
    best_X_imputed = None
    best_state_dict = None

    for pre_epoch in range(epochs):
        gnn.train()
        gnn2.train()
        regressor.train()
        opt.zero_grad()
        loss = 0
        X_imputed = copy.copy(X)

        # edge_index = None
        for i in range(graph_impute_layers):
            if args.dynamic == 'true':
                X_emb = model_list[i](X_imputed)
            else:
                X_emb, edge_index = model_list[i](X_imputed, edge_index)
            print(i, 'X_emb shape:', X_emb.shape)
            print(i, 'X_emd:', X_emb)

            # X_emb = F.relu(X_emb)
            pred = regressor(X_emb)
            X_imputed = X*X_mask + pred*(1 - X_mask)
            temp_loss = torch.sum(torch.abs(X - pred) * X_mask) / (torch.sum(X_mask) + 1e-5)
            # print('temp loss:', temp_loss.item())
            loss += temp_loss

        loss.backward()
        opt.step()
        train_loss = loss.item()
        print('{n} epoch loss:'.format(n=pre_epoch), train_loss)

        # trans_X = X_imputed * std_f + mean_f
        trans_X = copy.copy(X_imputed)
        # trans_eval_X = eval_X * std_f + mean_f
        trans_eval_X = copy.copy(eval_X)
        print('trans_X shape', trans_X.shape)
        print('trans_eval_X shape', trans_eval_X.shape)

        epoch_state_dict = {'gnn': gnn.state_dict(), 'gnn2': gnn2.state_dict(),  'regressor': regressor.state_dict()}
        # state_dict_list.append(epoch_state_dict)

        gnn.eval()
        gnn2.eval()
        regressor.eval()

        with torch.no_grad():
            # mae_error = torch.sum(torch.abs(trans_X - trans_eval_X) * eval_mask)/torch.sum(eval_mask)

            # mse_error = F.mse_loss(trans_X[torch.where(eval_mask == 1)],
            #                                    trans_eval_X[torch.where(eval_mask == 1)])

            mae_error = cal_mae(trans_X, trans_eval_X, eval_mask)

            mse_error = cal_mse(trans_X, trans_eval_X, eval_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

            mape_error = cal_mre(trans_X, trans_eval_X, eval_mask)

            # mae_error_mape = torch.sum(torch.abs((trans_X[torch.where(eval_mask == 1)]-
            #                                    trans_eval_X[torch.where(eval_mask == 1)])/trans_eval_X[torch.where(eval_mask == 1)]))/torch.sum(eval_mask)


            print('{epcoh}:valid impute value samples:'.format(epcoh=pre_epoch), (trans_X[torch.where(eval_mask == 1)]))
            print('valid true value samples:', (trans_eval_X[torch.where(eval_mask == 1)]))

            print('valid impute error MAE:', mae_error.item())
            print('valid impute error MSE:', mse_error.item())
            print('valid impute error MRE:', mape_error.item())

            # mae_error_list.append(round(mae_error.item(), 6))
            # print('valid min impute error MAE:', min(mae_error_list))

            if mae_error.item() < min_mae_error:
                opt_epoch = copy.copy(pre_epoch)
                min_mae_error = round(mae_error.item(), 6)
                print('{epoch}_opt_mae_error'.format(epoch=pre_epoch), min_mae_error)

                min_mse_error = round(mse_error.item(), 6)
                min_mape_error = round(mape_error.item(), 6)

                opt_time = (datetime.now()-st).total_seconds()/60
                opt_time = round(opt_time, 6)
                print('{epoch}_opt time:'.format(epoch=pre_epoch), opt_time)

                best_X_imputed = copy.copy(X_imputed)
                best_X_imputed = best_X_imputed.data.cpu().numpy()[-ori_X_row:]
                best_X_imputed = best_X_imputed*(1-ori_X_mask) + ori_X*ori_X_mask
                output_best_X_imputed = best_X_imputed*std_X + mean_X
                output_best_X_imputed = np.round(output_best_X_imputed, 4)

                print('output_best_X_imputed shp', output_best_X_imputed.shape)
                best_state_dict = copy.copy(epoch_state_dict)

    et = datetime.now()
    total_elapsed_time = round((et-st).total_seconds()/60, 6)
    results_list = [opt_epoch, min_mae_error, min_mse_error, min_mape_error, num_of_params, model_size, opt_time, total_elapsed_time]

    # best_X_imputed = X_imputed_list[arg_min_mae_error]
    # best_X_imputed = best_X_imputed.data.cpu().numpy().tolist()[-ori_X_row:]

    # best_state_dict = state_dict_list[arg_min_mae_error]

    if mae_last and (min_mae_error > mae_last) and (state == 'true'):
        best_state_dict = copy.copy(initial_state_dict)

    return best_state_dict, keep_X.tolist(), keep_mask, results_list, min_mae_error, output_best_X_imputed, ori_X_mask

incre_mode = args.incre_mode # 'alone',  'data', 'state', 'state+transfer', 'data+state', 'data+state+transfer'
# sample_ratio = args.sample_ratio
prefix = args.prefix

if args.dataset == 'KDM':
    num_windows = int(60/args.window)
elif args.dataset == 'ICU':
    num_windows = int(48/args.window)
elif args.dataset == 'Airquality':
    num_windows = int(24/args.window)
elif args.dataset == 'LHS':
    num_windows = int(200/args.window)


results_schema = ['opt_epoch', 'opt_mae', 'mse', 'mape', 'para', 'memo', 'opt_time', 'tot_time']

num_of_iteration = args.num_of_iter
iter_results_list = []
iter_imputation_list = []
for iteration in range(num_of_iteration):
    results_collect = []
    imputation_collect =  []
    for w in range(num_windows):
        print(f'which time window:{w}')
        if w == 0:
            window_best_state, X_last, mask_last, window_results, mae_last, output_best_X_imputed, ori_X_mask = window_imputation(start=w, end=w+1, sample_ratio=1/float(num_windows))
            results_collect.append(window_results)
            imputation_collect.append(output_best_X_imputed)
        else:
            if incre_mode == 'alone':
                window_best_state, X_last, mask_last, window_results, mae_last, output_best_X_imputed, ori_X_mask = window_imputation(start=w, end=w+1, sample_ratio=1/num_windows)
                results_collect.append(window_results)
                imputation_collect.append(output_best_X_imputed)

            elif incre_mode == 'data':
                window_best_state, X_last, mask_last, window_results, mae_last, output_best_X_imputed, ori_X_mask = window_imputation(start=w, end=w + 1, sample_ratio=1/num_windows, X_last=X_last, mask_last=mask_last)
                results_collect.append(window_results)
                imputation_collect.append(output_best_X_imputed)

            elif incre_mode == 'state':
                window_best_state, X_last, mask_last, window_results, mae_last, output_best_X_imputed, ori_X_mask = window_imputation(start=w, end=w + 1, sample_ratio=1/num_windows,initial_state_dict=window_best_state, mae_last=mae_last)
                results_collect.append(window_results)
                imputation_collect.append(output_best_X_imputed)

            elif incre_mode == 'state+transfer':
                window_best_state, X_last, mask_last, window_results, mae_last, output_best_X_imputed, ori_X_mask = window_imputation(start=w, end=w + 1, sample_ratio=1/num_windows,initial_state_dict=window_best_state, transfer=True, mae_last=mae_last)
                results_collect.append(window_results)
                imputation_collect.append(output_best_X_imputed)

            elif incre_mode == 'data+state':
                window_best_state, X_last, mask_last, window_results, mae_last, output_best_X_imputed, ori_X_mask = window_imputation(start=w, end=w+1, sample_ratio=1/num_windows, initial_state_dict=window_best_state, X_last=X_last, mask_last=mask_last, mae_last=mae_last)
                results_collect.append(window_results)
                imputation_collect.append(output_best_X_imputed)

            elif incre_mode == 'data+state+transfer':
                window_best_state, X_last, mask_last, window_results, mae_last, output_best_X_imputed, ori_X_mask = window_imputation(start=w, end=w+1, sample_ratio=1/num_windows, initial_state_dict=window_best_state, X_last=X_last, mask_last=mask_last, transfer=True, mae_last=mae_last)
                results_collect.append(window_results)
                imputation_collect.append(output_best_X_imputed)

    # output_best_X_imputed = np.round(output_best_X_imputed.astype(float), 4)
    # np.savetxt(f'./iif_exp/{iteration}_{args.dataset}_{args.method}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}_Mask.txt', ori_X_mask, fmt='%.0f')

    df = pd.DataFrame(results_collect, index=range(num_windows), columns=results_schema)
    iter_results_list.append(df)
    iter_imputation_list.append(output_best_X_imputed)

print('ready to write data!')
# avg_df = sum(iter_results_list)/num_of_iteration
# avg_df = avg_df.round(4)


avg_X_imputed = sum(iter_imputation_list)/num_of_iteration
if args.dataset == 'KDM':
    output_X_Y = np.concatenate([avg_X_imputed, base_Y], axis=1)
    np.savetxt(f'./iif_exp/APP_{args.prefix}_{args.dataset}_{args.method}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}_imputation_XY.txt', output_X_Y, fmt='%.2f')
elif args.dataset == 'ICU':
    np.savetxt(f'./iif_exp/APP_{args.prefix}_{args.dataset}_{args.method}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}_impute_X.txt', avg_X_imputed, fmt='%.2f')
    np.savetxt(f'./iif_exp/APP_{args.prefix}_{args.dataset}_{args.method}_window_{args.window}_eval_{args.eval_ratio}_stream_{args.stream}_impute_Y.txt', base_Y, fmt='%.2f')





# avg_df.to_csv(f'./iif_exp/{args.prefix}_{args.dataset}_{args.method}_{args.incre_mode}_{args.k}_window_{args.window}_epoch_{args.epochs}_eval_{args.eval_ratio}_stream_{args.stream}'
#               f'_state_{args.state}_thre_{args.thre}.csv', header=True, index=True)


# if args.prefix == 'testNumStream':
#     avg_df.to_csv(f'./exp_results/{args.method}_{args.prefix}_{args.dataset}_{args.k}_incre_{args.incre_mode}_window_{args.window}_epoch_{args.epochs}_eval_{args.eval_ratio}_stream_{args.stream}.csv',header=True, index=True)
# else:
#     avg_df.to_csv(f'./exp_results/{args.method}_{args.prefix}_{args.dataset}_{args.k}_incre_{args.incre_mode}_window_{args.window}_epoch_{args.epochs}_eval_{args.eval_ratio}.csv',header=True, index=True)
#
# print('done!')
# print('finishing time:', datetime.now())

# if args.prefix == 'testNumStream':
#     avg_df.to_csv(f'./iif_exp/{args.prefix}_{args.dataset}_{args.k}_{args.base}_incre_{args.incre_mode}_window_{args.window}_epoch_{args.epochs}_eval_{args.eval_ratio}_stream_{args.stream}.csv', header=True, index=True)
# else:
#     avg_df.to_csv(f'./iif_exp/{args.prefix}_{args.dataset}_{args.k}_{args.base}_incre_{args.incre_mode}_window_{args.window}_epoch_{args.epochs}_eval_{args.eval_ratio}.csv', header=True, index=True)
print('done!')
print('finishing time:', datetime.now())































