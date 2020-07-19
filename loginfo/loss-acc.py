# -*- coding: cp936 -*-
f_out=["webkb_cpu_base.out","webkb_cpu_large.out",\
       "webkb_gpu_base.out","webkb_gpu_large.out",\
       "r52_cpu_base.out","r52_cpu_large.out",\
       "r52_gpu_base.out","r52_gpu_large.out"]
f_jpg=["webkb_cpu_base.jpg","webkb_cpu_large.jpg",\
       "webkb_gpu_base.jpg","webkb_gpu_large.jpg",\
       "r52_cpu_base.jpg","r52_cpu_large.jpg",\
       "r52_gpu_base.jpg","r52_gpu_large.jpg"]
f_num=[60,60,60,60,116,116,116,116]
f_step=[15,15,15,15,29,29,29,29]

for k in range(8):
   f=open(f_out[k],"r")
   loss=[]
   acc=[]
   x=[i for i in range(f_num[k])]

   for i in range(f_num[k]):
      s=f.readline()
      l=len(s)
      lossi=eval(s[-32:-26])
      acci=eval(s[-6:-1])
      loss.append(lossi)
      acc.append(acci)


   import matplotlib.pyplot as plt
   import numpy as np 

   fig,ax=plt.subplots()

   x_ticks=np.arange(0,f_num[k],f_step[k])
   ax.set_xticks(x_ticks)

   x_labels=[0,1,2,3]
   ax.set_xticklabels(x_labels)

   plt.plot(x,loss,label='loss',color='red')
   plt.ylabel('loss')
   plt.legend(loc=2)
   plt.xlabel('epoch')

   plt.twinx()

   plt.plot(x,acc,label='acc',color='blue')
   plt.ylabel('acc')
   plt.legend(loc=1)

   title='loss-acc OF '+f_out[k]
   plt.title(title)
   
   plt.savefig(f_jpg[k])
