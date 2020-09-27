'''torchtext常见API一览

torchtext.data.Example : 用来表示一个样本，数据和标签
torchtext.vocab.Vocab: 词汇表，可以导入一些预训练词向量
torchtext.data.Datasets: 数据集类，__getitem__返回 Example实例, torchtext.data.TabularDataset是其子类。
torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）创建 Example时的 预处理，batch 时的一些处理操作。
torchtext.data.Iterator: 迭代器，用来生成 batch
torchtext.datasets: 包含了常见的数据集.
'''

if __name__ == '__main__':
    import torch
    import string, re
    import torchtext

    MAX_WORDS = 10000 # 保留1w个高频词
    MAX_LEN = 200# 每个样本保留200个词的长度
    BATCH_SZIE   = 20
    # 分词方式 空格分词，标点分开
    tokenizer = lambda x: re.sub('[%s]' % string.punctuation, "", x).split(" ")


    # 过滤掉低频词
    def filterLowFreqWords(arr, vocab):
        arr = [[x if x < MAX_WORDS else 0 for x in example]
               for example in arr]
        return arr


    # 1,定义各个字段的预处理方法
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True,
                                fix_length=MAX_LEN, postprocessing=filterLowFreqWords)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    #2,构建表格型dataset
    #torchtext.data.TabularDataset可读取csv,tsv,json等格式
    ds_train, ds_test = torchtext.data.TabularDataset.splits(
            path='./data/imdb', train='train.tsv',test='test.tsv', format='tsv',
            fields=[('label', LABEL), ('text', TEXT)],skip_header = False)