# Lottery Ticket Hypothesis on ImageNet

## Results

__Iterative magnitude pruning with weight rewinding.__
To establish a rewind point, we begin by training the model from a random initialization for 16k iterations. During this training phase, we employ the SGD optimizer with a momentum of 0.9, weight decay of 0.0001, a mini-batch size of 2048, and a constant learning rate of 0.8. Afterwards, we employ a cosine decaying learning rate schedule over 48k iterations in each cycle of the IMP algorithm. During this iterative pruning phase, we employ a fixed pruning ratio of 0.8, meaning that 80% of the existing weights are retained while the remaining 20% are pruned. All training runs are performed using mixed precision training on eight TPUv3 cores.1

__Pruning convolutional weights.__
We focus solely on applying pruning to the convolutional layer, while leaving the batch normalization and fully-connected layers unpruned. This means that the batch normalization layers consistently have 53,120 affine parameters and 53,120 buffers. As for the fully-connected layer, it comprises 2,048,000 parameters.

| Cycle | # Params (Conv)   | IN    | IN-V2 | IN-R  | IN-A  | IN-S  |
| :-:   | -:                | :-:   | :-:   | :-:   | :-:   | :-:   |
| 000   | 23454912 (1.0000) | 76.06 | 63.69 | 35.50 | 1.933 | 23.85 |
| 001   | 18764028 (0.8000) | 76.33 | 64.11 | 35.29 | 1.960 | 23.91 |
| 002   | 15011282 (0.6400) | 76.45 | 63.88 | 35.33 | 2.107 | 24.27 |
| 003   | 12009076 (0.5120) | 76.48 | 64.04 | 35.30 | 2.387 | 24.11 |
| 004   |  9607294 (0.4096) | 76.58 | 64.02 | 35.09 | 2.373 | 23.82 |
| 005   |  7685860 (0.3277) | 76.48 | 63.83 | 35.00 | 2.520 | 23.52 |
| 006   |  6148710 (0.2622) | 76.53 | 63.83 | 34.97 | 2.480 | 23.89 |
| 007   |  4918983 (0.2097) | 76.54 | 64.14 | 34.89 | 2.307 | 23.52 |
| 008   |  3935188 (0.1678) | 76.23 | 63.69 | 34.53 | 2.240 | 23.08 |
| 009   |  3148156 (0.1342) | 76.13 | 63.60 | 34.17 | 2.227 | 22.90 |
| 010   |  2518532 (0.1074) | 75.87 | 63.19 | 34.19 | 2.093 | 22.92 |
| 011   |  2014828 (0.0859) | 75.53 | 62.83 | 33.82 | 1.987 | 22.52 |
| 012   |  1611863 (0.0687) | 75.21 | 62.58 | 33.29 | 1.813 | 21.95 |
| 013   |  1289491 (0.0550) | 74.77 | 62.08 | 32.60 | 1.613 | 21.37 |
| 014   |  1031592 (0.0440) | 74.00 | 61.40 | 31.98 | 1.627 | 20.86 |
| 015   |   825273 (0.0352) | 73.36 | 60.57 | 31.71 | 1.680 | 20.39 |
| 016   |   660219 (0.0281) | 72.62 | 59.25 | 31.22 | 1.413 | 19.92 |
| 017   |   528175 (0.0225) | 71.51 | 58.63 | 30.74 | 1.413 | 19.21 |
| 018   |   422542 (0.0180) | 70.52 | 58.03 | 30.08 | 1.333 | 18.57 |
