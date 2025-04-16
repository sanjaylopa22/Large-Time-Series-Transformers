# LSTSMs (Large Time Series Models for Time Series Forecasting)

Large time-series models, pre-training datasets, and adaptation techniques (Future Features).

## Model Checklist

- [x] **Moirai** - Unified Training of Universal Time Series Forecasting Transformers. [[ICML 2024]](https://arxiv.org/abs/2402.02592), [[Code]](https://github.com/SalesforceAIResearch/uni2ts)
- [x] **Moment** - MOMENT: A Family of Open Time-series Foundation Model. [[ICML 2024]](https://arxiv.org/abs/2402.03885), [[Code]](https://github.com/moment-timeseries-foundation-model/moment)
- [x] **Timer** - Timer: Generative Pre-trained Transformers Are Large Time Series Models. [[ICML 2024]](https://arxiv.org/abs/2402.02368), [[Code]](https://github.com/thuml/Large-Time-Series-Model)
- [x] **Timer-XL** - Timer-XL: Long-Context Transformer for Unified Time Series Forecasting. [[arxiv 2024]](https://arxiv.org/abs/2410.04803), [[Code]](https://github.com/thuml/Timer-XL)
- [x] **GPT4TS** - One Fits All: Power General Time Series Analysis by Pretrained LM. [[arxiv 2024]](https://arxiv.org/abs/2302.11939), [[Code]](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)


## Usage

1. Install Python 3.10. For convenience, execute the following command.

```
pip install -r requirements.txt
```

1. Place downloaded data in the folder ```./dataset```. Here is a [dataset summary](./figures/datasets.png).

- For pre-training:
  * [UTSD](https://huggingface.co/datasets/thuml/UTSD) contains 1 billiion time points for large-scale pre-training (in numpy format): [[Download]](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
).
  * [ERA5-Familiy](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) (40-year span, thousands of variables) for domain-specific model: [[Download]](https://cloud.tsinghua.edu.cn/f/7fe0b95032c64d39bc4a/).

- For superwised training or modeling adaptation
  * Datasets from [TSLib](https://github.com/thuml/Time-Series-Library) : [[Download]](https://cloud.tsinghua.edu.cn/f/4d83223ad71047e28aec/).

2. We provide pre-training and adaptation scripts under the folder `./scripts/`. You can conduct experiments using the following examples:

```
# Supervised training
# (a) one-for-one forecasting
bash ./scripts/supervised/forecast/moirai_ecl.sh
# (b) one-for-all (rolling) forecasting
bash ./scripts/supervised/rolling_forecast/timer_xl_ecl.sh

# Large-scale pre-training
# (a) pre-training on UTSD
bash ./scripts/pretrain/timer_xl_utsd.sh
# (b) pre-training on ERA5
bash ./scripts/pretrain/timer_xl_era5.sh

# Model adaptation
# (a) full-shot fine-tune
bash ./scripts/adaptation/full_shot/timer_xl_etth1.sh
# (b) few-shot fine-tune
bash ./scripts/adaptation/few_shot/timer_xl_etth1.sh
```

3. Develop your large time-series model.

- Add the model file to the folder `./models`. You can follow the `./models/timer_xl.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

4. Or evaluate the zero-shot performance of large time-series models. Here we list some resources:
- Chronos: https://huggingface.co/amazon/chronos-t5-base
- Moirai: https://huggingface.co/Salesforce/moirai-1.0-R-base
- Timer-XL: https://huggingface.co/thuml/timer-base-84m

> [!NOTE]
> LTMs are still small in scale compared to large models of other modalities. We prefer to include and implement models requiring affordable training resources as efficiently as possible (for example, using several NVIDIA RTX 3090s or A100s).

## Acknowledgment

We appreciate the following GitHub repos a lot for their valuable code and efforts:
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- Large-Time-Series-Model (https://github.com/thuml/Large-Time-Series-Model)
