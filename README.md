# Introduction
This is the repository for MPIN project.

# Requirements
- Pytorch 1.8.1
- Numpy 1.19.2
- Pandas 1.1.3
- Sklearn 0.24.1
- tsdb 
- pypots
- torch_geometric

You may use " pip3 install -r requirements.txt" to install the above libraries.

# Usage

**Snapshot imputation for a window**: 

``` 
cd ./snapshot for a window;
```
***MPIN***

``` 
bash run_graph_imputer_incre_27.sh alone 4 0.1 200 SAGE ICU 10 testMissRate 1;
``` 

***baselines***

``` 
bash run_graph_imputer_incre_28.sh alone 4 0.5 40 ICU 10 testMissRate 1; ### FP

bash run_graph_imputer_incre_29.sh alone 4 0.5 ICU 10 testMissRate MICE 1; ### traditional imputers, e.g., MICE

bash run_saits_exp_app.sh 4 0.5 ICU testMissRate saits 1; ### neural network based imputers, e.g., saits

``` 


**Continuous imputation**:

``` 
cd ./continuous;
bash run_incre_imputation_6.sh data 4 0.5 200 SAGE ICU 10 testMissRate 1.0 true 0.6;

```

# Explaination of Parameters

**Snapshot imputation for a window**: 

e.g, bash run_graph_imputer_incre_27.sh alone 4 0.5 200 SAGE ICU 10 testMissRate 1

4：window length
0.5: missing ratio
200: training epochs
SAGE: base model, other options such as GAT, GCN
ICU: dataset, other options such as Airquality, KDM (i.e.,Wi-Fi).
10: K value of KNN
testMissRate: effect, other options such as testWindowLen, testNumStream 
1: ratio of streams

**Continuous imputation**:

bash run_incre_imputation_6.sh data 4 0.5 200 SAGE ICU 10 testNumStream 1.0 true 0.6

data: update mechanism MPIN-D, other options such as state (MPIN-M), data+state (MPIN-DM), alone (MPIN-P)
4：the same meaning as above
0.5: the same meaning as above 
200: the same meaning as above
SAGE: ...


# Acknowledgements

We appreciate the work of SAITS, and their contributed codes available in [here](https://github.com/WenjieDu/SAITS).



