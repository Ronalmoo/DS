# winter_plan
winter_vacation plan
# implementation

### Classification
+ Using the [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)
+ Hyper-parameter was arbitrarily selected.

+ Tokenizer: Okt
|                  | Train ACC | Validation ACC | Test ACC |
| :--------------- | :-------: | :------------: | :------: |
| Baseline (Feed Forward)         |  -  |     -     |  -  |
| LSTM           |  88.917%  | 84.389% | 84.189% |
| Bi-LSTM          | 97.09% | 96.93% | - |
| Transformer          | - | - | - |

+ Tokenizer: Mecab
|                  | Train ACC | Validation ACC | Test ACC |
| :--------------- | :-------: | :------------: | :------: |
| Baseline (Feed Forward)         |  -  |     -     |  -  |
| LSTM           |  89.271%  | 85.330% | 84.256% |
| Bi-LSTM          | 97.09% | 96.93% | - |
| Transformer          | - | - | - |

+ Tokenizer: Sentencepiece
|                  | Train ACC | Validation ACC | Test ACC |
| :--------------- | :-------: | :------------: | :------: |
| Baseline (Feed Forward)         |  -  |     -     |  -  |
| LSTM           |  92.672%  | 84% | 83.559% |
| Bi-LSTM          | 97.09% | 96.93% | - |
| Transformer          | - | - | - |
