nohup python3 -u nn.py  --window $1 --eval_ratio $2 --dataset $3 --prefix $4 --method $5 --stream $6  > ./log/$5_$4_$3_window_$1_eval_$2_stream_$6.log 2>&1 &
