import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
from bert import data
import pandas as pd
import os, sys
import numpy as np
from tkinter import *
import tkinter.messagebox
from random import random
from math import sqrt
import time
import random

root = Tk()
root.wm_title('myWindow')

all_labels = ["student", "course", "faculty", "project"]

def _quit():
    root.quit()
    root.destroy()

def p1():
   toy = str_sentence.get()
   all_labels = ["student", "course", "faculty", "project"]
   data0 = [[all_labels[0], toy]]
   data0 = pd.DataFrame(data0)
   data0.to_csv("toy0.tsv", index = False, sep = '\t')
   data1 = [[all_labels[1], toy]]
   data1 = pd.DataFrame(data0)
   data1.to_csv("toy1.tsv", index = False, sep = '\t')
   data2 = [[all_labels[2], toy]]
   data2 = pd.DataFrame(data0)
   data2.to_csv("toy2.tsv", index = False, sep = '\t')
   data3 = [[all_labels[3], toy]]
   data3 = pd.DataFrame(data0)
   data3.to_csv("toy3.tsv", index = False, sep = '\t')
   
   eval_metric = []
   start = time.time()
   eval_metric.append(predict(prefix, bert_testloader0, 0))
   eval_metric.append(predict(prefix, bert_testloader1, 0))
   eval_metric.append(predict(prefix, bert_testloader2, 0))
   eval_metric.append(predict(prefix, bert_testloader3, 0))
   end = time.time()
   predict_result = all_labels[eval_metric.index(max(eval_metric))]
   Timecost = end - start
   Timecost = round(Timecost,2)
   throughout_out = batch_size*(len(bert_testloader0)+len(bert_testloader1)+len(bert_testloader2)+len(bert_testloader3))/(end-start)
   throughout_out = round(throughout_out,2)
   
   result_predict.delete('0.0',END)
   result_predict.insert('1.0',str(predict_result))
   time_cost.delete('0.0',END)
   time_cost.insert(INSERT,str(Timecost)+"s")
   throughout.delete('0.0',END)
   throughout.insert(INSERT,str(throughout_out)+"samples/s")


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




#gui界面设计
root.geometry('390x220')
#sentence输入标签
label_1 = Label(root,text = 'Enter a sentence: ',font=('Times New Roman',9))
label_1.pack()
label_1.place(x = 20,y = 20,anchor = NW)
#sentence输入文本框
str_sentence = StringVar(value='')
entry_oid = Entry(root, textvariable=str_sentence)
entry_oid.pack()
entry_oid.place(in_=label_1,relx = 1.1)
#sentence确认按钮
button_oid = Button(root,text = "Confirm",command = p1)
button_oid.pack()
button_oid.place(height = 24,width = 50,x = 280,y = 20)
#输出预测标签
label_2 = Label(root,text = "Prediction: ",font=('Times New Roman',9))
label_2.pack()
label_2.place(x = 20,y = 60, anchor = 'nw')
#输出预测
result_predict = Text(root,height=1.5,width=20)
result_predict.pack()
result_predict.place(x = 115, y = 60)
#输出耗时
label_3 = Label(root,text = "Time cost: ",font=('Times New Roman',9))
label_3.pack()
label_3.place(x = 20,y = 100, anchor = 'nw')
time_cost = Text(root,height=1.5,width=20)
time_cost.pack()
time_cost.place(x = 115, y = 100)
#输出流量
label_4 = Label(root,text = "Throughout: ",font=('Times New Roman',9))
label_4.pack()
label_4.place(x = 20,y = 140, anchor = 'nw')
throughout = Text(root,height=1.5,width=20)
throughout.pack()
throughout.place(x = 115, y = 140)

root.mainloop()