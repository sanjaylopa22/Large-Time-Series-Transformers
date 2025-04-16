export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl
token_num=32
token_len=96
seq_len=$[$token_num*$token_len]

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/era5_pretrain/ \
  --data_path pretrain.npy \
  --model_id era5_pretrain \
  --model timer_xl \
  --data Era5_Pretrain  \
  --seq_len 3072 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 3072 \
  --test_pred_len 96 \
  --e_layers 4 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --dp \
  --devices 0            #--devices 0,1,2,3,4,5,6,7