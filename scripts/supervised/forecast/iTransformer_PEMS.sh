export CUDA_VISIBLE_DEVICES=0
model_name=moment
token_num=4
token_len=24
seq_len=$[$token_num*$token_len]
  
  
 python -u run.py  --task_name long_term_forecast  --is_training 1  --root_path ./dataset/PEMS08/  --data_path PEMS08.npz  --model_id PEMS08  --model iTransformer  --data PEMS  --features M  --seq_len 96  --label_len 0  --pred_len 24  --enc_in 170  --dec_in 170  --c_out 170  --des 'Exp'  --itr 1   --channel_independence 0  --d_model 128  --batch_size 16  --learning_rate 0.0001  --train_epochs 5  --patience 3  --down_sampling_layers 3  --down_sampling_method avg  --down_sampling_window 2


