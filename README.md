# Winter_plan
winter_vacation plan
# Implementation

### Classification
+ Using the [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)
+ Hyper-parameter was arbitrarily selected.


```
usage: main_last.py [-h] [--emb_size EMB_SIZE] [--n_layer N_LAYER]
                    [--batch_size BATCH_SIZE] [--hidden_size HIDDEN_SIZE]
                    [--output_size OUTPUT_SIZE] [--lr LR] [--epochs EPOCHS]
                    [--clip CLIP]

optional arguments:
  -h, --help            show this help message and exit
  --emb_size EMB_SIZE   Embedding size
  --n_layer N_LAYER     number of layer
  --batch_size BATCH_SIZE
                        batch size
  --hidden_size HIDDEN_SIZE
                        size of hidden layer
  --output_size OUTPUT_SIZE
                        size of output layer
  --lr LR               learning rate
  --epochs EPOCHS       epoch number of training
  --clip CLIP           learning rate
```
+ Tokenizer: Okt


|                  | Train ACC(120,000) | Validation ACC(30,000) | Test ACC(50,000) |
| :--------------- | :-------: | :------------: | :------: |
| CNN (Baseline)         |  -  |     -     |  -  |
| LSTM           |  87.649%  | 84.709% | 83.189% |
| Bi-LSTM          | 90.489 % | 83.119% | 82.385% |
| Transformer          | - | - | - |


+ Tokenizer: Mecab

|                  | Train ACC(120,000) | Validation ACC(30,000) | Test ACC(50,000) |
| :--------------- | :-------: | :------------: | :------: |
| CNN (Baseline)         | 86.516% | 86.330% | 85.857% |
| LSTM           |  90.085%  | 85.330% | 84.256% |
| Bi-LSTM          | 93.405% | 85.216% | 85.137% |
| Transformer          | - | - | - |


+ Tokenizer: Sentencepiece(SKT)

|                  | Train ACC(120,000) | Validation ACC(30,000) | Test ACC(50,000) |
| :--------------- | :-------: | :------------: | :------: |
| CNN (Baseline)         |  -  |     -     |  -  |
| LSTM           |  86.998%  | 83.443% | 82.021% |
| Bi-LSTM          | 87.3% | 83.243% | 82.611% |
| Transformer          | - | - | - |
