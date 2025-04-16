export CUDA_VISIBLE_DEVICES=0
model_name=timer
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]


  python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS08/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_Full_shot \
  --model timer \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 12 \
  --e_layers 4 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --train_epochs 5 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --adaptation \
  --pretrain_model_path checkpoints/timer_Era5_Pretrain/checkpoint.pth
  
  python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS08/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_Full_shot \
  --model timer \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 24 \
  --e_layers 4 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --train_epochs 5 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --adaptation \
  --pretrain_model_path checkpoints/timer_Era5_Pretrain/checkpoint.pth
  
  python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS08/ \
  --model_id PEMS08_Full_shot \
  --model timer \
  --data MultivariateDatasetBenchmark   \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 48 \
  --e_layers 4 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --train_epochs 5 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --adaptation \
  --pretrain_model_path checkpoints/timer_Era5_Pretrain/checkpoint.pth
  
