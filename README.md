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
| LSTM           |  88.917%  | 84.389% | - |
| Bi-LSTM          | 97.09% | 96.93% | - |
| Transformer          | - | - | - |

