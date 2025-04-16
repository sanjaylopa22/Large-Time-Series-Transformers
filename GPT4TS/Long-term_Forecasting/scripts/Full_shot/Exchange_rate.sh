
export CUDA_VISIBLE_DEVICES=0

seq_len=672
model=GPT4TS

for percent in 100
do
for pred_len in 12 24 48
do
for lr in 0.0001
do

python main.py \
    --root_path ./datasets/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id exchange_rate_GPT4TS'_'$gpt_layer'_'336'_'$pred_len'_'$percent \
    --data custom \
    --seq_len 96 \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 5 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1

done
done
done
