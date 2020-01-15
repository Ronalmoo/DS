import pandas as pd
import torchtext
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator
from konlpy.tag import Mecab


# train_df = pd.read_csv('ratings_train.txt', delimiter='\t')
# test_df = pd.read_csv('ratings_test.txt', delimiter='\t')
# print(len(train_df))
# print(len(test_df))


class MovieDataLoader(object):
    def __init__(self, batch_size=64, max_vocab=999999,
                 min_freq=1, tokenizer=Mecab(), shuffle=True):
        super().__init__()
        self.TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer.morphs,
                          lower=True, batch_first=True, fix_length=20)
        self.LABEL = Field(sequential=False, use_vocab=False, is_target=True)

        train_data, valid_data, test_data = TabularDataset.splits(path='data/', train='train.txt',
                                                                  validation='validation.txt',
                                                                  test='test.txt', format='tsv',
                                                                  fields=[('text', self.TEXT), ('label', self.LABEL)],
                                                                  skip_header=True)
        self.TEXT.build_vocab(train_data, max_size=max_vocab, min_freq=min_freq)
        self.LABEL.build_vocab(train_data)

        # self.train_loader = BucketIterator(dataset=train_data, batch_size=batch_size)
        # self.test_loader = BucketIterator(dataset=valid_data, batch_size=batch_size)
        self.tr_dl, self.val_dl, self.test_dl = BucketIterator.splits((train_data, valid_data, test_data),
                                                                       sort_key=lambda x: len(x.text), batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    dataset = MovieDataLoader()
    tr_dl, val_dl, test_dl = dataset.tr_dl, dataset.val_dl, dataset.test_dl
    print(len(tr_dl))
    print(len(val_dl))
    print(len(test_dl))
    vocab_size = len(dataset.TEXT.vocab)
    n_classes = len(dataset.LABEL.vocab)
    print(vocab_size)
    print(n_classes)
    print(dataset.TEXT.vocab.stoi)
    x, y = next(iter(tr_dl))
    print(x.size())
    print(y.size())
    print(len(tr_dl.dataset))

# tokenizer = MeCab()

# ID = Field(sequential=False, use_vocab=False)
# TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer.morphs,
#              lower=True, batch_first=True, fix_length=20)
# LABEL = Field(sequential=False, use_vocab=False, is_target=True)

# train_data, test_data = TabularDataset.splits(path='.', train='ratings_train.txt',
#                                               test='ratings_test.txt', format='tsv',
#                                               fields=[("id", ID), ('text', TEXT), ('label', LABEL)], skip_header=True)


# print(len(train_data))
# print(len(test_data))
# print(vars(train_data[0]))

# TEXT.build_vocab(train_data, min_freq=10, max_size=999999)
# print(len(TEXT.vocab))
# print(TEXT.vocab.stoi)

# batch_size = 5
# train_loader = Iterator(dataset=train_data, batch_size=batch_size)
# test_loader = Iterator(dataset=test_data, batch_size=batch_size)

# print(len(train_loader))
# print(len(test_loader))
# batch = next(iter(train_loader))
# print(batch.text)