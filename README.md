# ResMatch Implementation

PyTorch implementation of ["ResMatch: Residual Attention Learning for Feature Matching"](https://arxiv.org/abs/2307.05180), by Yuxin Deng and Jiayi Ma.

## News
08-2023 We upload new pretrained models to match new features, including [DISK](https://github.com/cvlab-epfl/disk)(NIPS2020), [ALIKED](https://github.com/Shiaoming/ALIKED)(TMM2022,TIM2023), [AWDesc](https://github.com/vignywang/AWDesc)(TPAMI2023). Performance on ScanNet is below:

|           | 5     | 10    | 15    | 20    | 5     | 10    | 15    | 20    | MS    | P     |
| --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| DISK       | 32.0  | 42.0  | 48.8 | 53.8  | 13.4  | 28.5  | 38.2  | 45.1  | 14.22 | 48.02 |
| ALIKED        | 37.1 | 47.8 | 54.5 | 59.1 | 16.4 | 33.1 | 43.4 | 50.2 | 14.51 | 48.80 |
| AWDesc     | 42.9 | 54.3 | 60.8 | 65.4 | 18.7 | 37.7 | 48.6 | 55.6 | 11.97 | 48.19 |


07-2023 We upload a pre-release version composite of models, pre-trained weights. It might be enough to repoduce the results in the codebase of [SGMNet](https://github.com/vdvchen/SGMNet). It will be a long time for the release of full codes since the paper is under review.
