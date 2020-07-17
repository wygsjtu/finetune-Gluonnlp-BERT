import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
from bert import data
import pandas as pd

toy = input("Enter a sentence: ")
all_labels = ["student", "course", "faculty", "project"]
data0 = [[all_labels[0], toy]]
data0 = pd.DataFrame(data0)
data0.to_csv("toy0.tsv", index = False, sep = '\t')
data1 = [[all_labels[1], toy]]
data1 = pd.DataFrame(data1)
data1.to_csv("toy1.tsv", index = False, sep = '\t')
data2 = [[all_labels[2], toy]]
data2 = pd.DataFrame(data2)
data2.to_csv("toy2.tsv", index = False, sep = '\t')
data3 = [[all_labels[3], toy]]
data3 = pd.DataFrame(data3)
data3.to_csv("toy3.tsv", index = False, sep = '\t')

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.cpu()

batch_size = 1
log_interval = 1
bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
num_discard_samples = 1
field_separator = nlp.data.Splitter('\t')
field_indices = [1, 0]
bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
max_len = 128
transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=False)
												
data_test_raw0 = nlp.data.TSVDataset(filename='toy0.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)
data_test0 = data_test_raw0.transform(transform)
test_sampler0 = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_test0],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_testloader0 = mx.gluon.data.DataLoader(data_test0, batch_sampler=test_sampler0)

data_test_raw1 = nlp.data.TSVDataset(filename='toy1.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)
data_test1 = data_test_raw0.transform(transform)
test_sampler1 = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_test1],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_testloader1 = mx.gluon.data.DataLoader(data_test0, batch_sampler=test_sampler1)

data_test_raw2 = nlp.data.TSVDataset(filename='toy2.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)
data_test2 = data_test_raw2.transform(transform)
test_sampler2 = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_test2],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_testloader2 = mx.gluon.data.DataLoader(data_test2, batch_sampler=test_sampler2)
data_test_raw3 = nlp.data.TSVDataset(filename='toy3.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)
data_test3 = data_test_raw3.transform(transform)
test_sampler3 = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_test3],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_testloader3 = mx.gluon.data.DataLoader(data_test3, batch_sampler=test_sampler3)

prefix = './model_bert_webkb'
metric = mx.metric.Accuracy()
def predict(prefix, dataloader, epoch_id):
    print('load symbol file directly as SymbolBlock for model deployment.')
    static_net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(prefix),
                                    ['data0', 'data1', 'data2'],
                                    '{}-000{}.params'.format(prefix, epoch_id))
    static_net.hybridize(static_alloc=True, static_shape = True)
    for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(dataloader):
            token_ids = token_ids.as_in_context(mx.cpu())
            valid_length = valid_length.as_in_context(mx.cpu())
            segment_ids = segment_ids.as_in_context(mx.cpu())
            label = label.as_in_context(mx.cpu())
            out = static_net(token_ids, segment_ids, valid_length.astype('float32'))
            metric.update([label], [out])
    return metric.get()[1]
eval_metric = []
eval_metric.append(predict(prefix, bert_testloader0, 0))
eval_metric.append(predict(prefix, bert_testloader1, 0))
eval_metric.append(predict(prefix, bert_testloader2, 0))
eval_metric.append(predict(prefix, bert_testloader3, 0))
print(all_labels[eval_metric.index(max(eval_metric))])