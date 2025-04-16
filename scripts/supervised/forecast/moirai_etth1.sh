export CUDA_VISIBLE_DEVICES=0
model_name=moirai
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity \
  --model moirai \
  --data MultivariateDatasetBenchmark  \
  --seq_len 672 \
  --input_token_len 96 \
  --output_token_len 720 \
  --test_seq_len 672 \
  --test_pred_len 720 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
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
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic \
  --model moirai \
  --data MultivariateDatasetBenchmark  \
  --seq_len 672 \
  --input_token_len 96 \
  --output_token_len 192 \
  --test_seq_len 672 \
  --test_pred_len 192 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --nonautoregressive
