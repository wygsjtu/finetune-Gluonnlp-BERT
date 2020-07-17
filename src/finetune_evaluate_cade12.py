# -*- coding: utf-8 -*- 
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
from bert import data
import time

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.gpu(0)

bert_base, vocabulary = nlp.model.get_model('bert_24_1024_16',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)

bert_classifier = nlp.model.BERTClassifier(bert_base, num_classes = 12, dropout=0.1)
# only need to initialize the classifier layer.
bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
bert_classifier.hybridize(static_alloc=True)

# softmax cross entropy loss for classification
loss_function = mx.gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

metric = mx.metric.Accuracy()

num_discard_samples = 0
field_separator = nlp.data.Splitter('\t')

field_indices = [1,0]
data_train_raw = nlp.data.TSVDataset(filename='cade-train-stemmed.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)
								 
# Use the vocabulary from pre-trained model for tokenization
bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

# The maximum length of an input sequence
max_len = 128

all_labels = ["01_servicos", "02_sociedade", "03_lazer", "04_informatica", "05_saude", "06_educacao", "07_internet", "08_cultura", "09_esportes", "10_noticias", "11_ciencias", "12_compras_on_line"]
transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=False)
data_train = data_train_raw.transform(transform)
data_test_raw = nlp.data.TSVDataset(filename='cade-test-stemmed.tsv',
                                 field_separator=field_separator,
                                 num_discard_samples=num_discard_samples,
                                 field_indices=field_indices)
data_test = data_test_raw.transform(transform)

# The hyperparameters
batch_size = 16
lr = 5e-6

# The FixedBucketSampler and the DataLoader for making the mini-batches
train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_train],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

trainer = mx.gluon.Trainer(bert_classifier.collect_params(), 'adam',
                           {'learning_rate': lr, 'epsilon': 1e-9})

params = [p for p in bert_classifier.collect_params().values() if p.grad_req != 'null']
grad_clip = 1

# Training the model with only three epochs
log_interval = 4
num_epochs = 4
for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(bert_dataloader):
        with mx.autograd.record():

            # Load the data to the GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # Forward computation
            out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()

        # And backwards computation
        ls.backward()

        # Gradient clipping
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(params, 1)
        trainer.update(1)

        step_loss += ls.asscalar()
        metric.update([label], [out])

        # Printing vital information
        if (batch_id + 1) % (log_interval) == 0:
            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                         .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, metric.get()[1]))
            step_loss = 0
test_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_test],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_testloader = mx.gluon.data.DataLoader(data_test, batch_sampler=test_sampler)
prefix = './model_bert_r52_large'

def deployment(net, prefix, dataloader):
    net.export(prefix, epoch = 0)
    print('Saving model at ', prefix)
    print('load symbol file directly as SymbolBlock for model deployment.')
    static_net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(prefix),
                                    ['data0', 'data1', 'data2'],
                                    '{}-0000.params'.format(prefix), ctx = ctx)
    static_net.hybridize(static_alloc=True, static_shape=True)
    for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(dataloader):
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            out = static_net(token_ids, segment_ids, valid_length.astype('float32'))
            metric.update([label], [out])

            # Printing vital information
            if (batch_id + 1) % (log_interval) == 0:
                print('[Batch {}/{}], acc = {:.3f}'
                            .format(batch_id + 1, len(dataloader),
                                    metric.get()[1]))
    return metric
tic = time.time()
eval_metric = deployment(bert_classifier, prefix, bert_testloader)
toc = time.time()
print(eval_metric)
print('Time cost = {:.2f} s, throughput = {:.2f} samples/s'.format(toc-tic,
                 batch_size*len(bert_testloader)/(toc - tic)))