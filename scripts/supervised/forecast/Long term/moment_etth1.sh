export CUDA_VISIBLE_DEVICES=0
model_name=moment
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model moment \
  --data MultivariateDatasetBenchmark  \
  --seq_len 672 \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len 672 \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --nonautoregressive


  
  python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange \
  --model moment \
  --data MultivariateDatasetBenchmark  \
  --seq_len 672 \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len 672 \
  --test_pred_len 192 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --nonautoregressive
   

  python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange \
  --model moment \
  --data MultivariateDatasetBenchmark  \
  --seq_len 672 \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len 672 \
  --test_pred_len 336 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --nonautoregressive
  
  
    python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange \
  --model moment \
  --data MultivariateDatasetBenchmark  \
  --seq_len 672 \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len 672 \
  --test_pred_len 720 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --nonautoregressive

