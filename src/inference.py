import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
from bert import data

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.cpu()

bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
                                             

num_discard_samples = 0
# Split fields by tabs
field_separator = nlp.data.Splitter('\t')
field_indices = [1,0]
bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
max_len = 128
all_labels = ["student", "course", "faculty", "project"]
batch_size = 4

transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=False)

data_test_raw = nlp.data.TSVDataset(filename='test.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)
data_test = data_test_raw.transform(transform)

test_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_test],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_testloader = mx.gluon.data.DataLoader(data_test, batch_sampler=test_sampler)

prefix = './model'
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
eval_metric.append(predict(prefix, bert_testloader, 0))
eval_metric.append(predict(prefix, bert_testloader, 1))
eval_metric.append(predict(prefix, bert_testloader, 2))
print(eval_metric)