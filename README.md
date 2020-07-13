# finetune-Gluonnlp-BERT
Course project of EE398 in SJTU.
## Requirements
- Python 3.6 (Anaconda 3.5.1 specifically)
- mxnet==1.6.0
- gluonnlp==0.9.2
- [BERT source code](https://gluon-nlp.mxnet.io/model_zoo/bert/index.html) 
- The BERT source code is required to be downloaded and unzipped into your own Anaconda or Python3 /Lib/site-packages folder, which is where the package manager such as PIP or Conda downloads it, or into the same folder as the function packages in the virtual environment in which you configure your code to execute. If you do not perform this step or put the Bert folder in the same directory as the script, an error may be reported after import.
## Code Structure
- The source code is in the/src folder of the repository, where the scripts [finetune_evaluate.py](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/finetune_evaluate.py) and [finetune_evaluate_r52.py](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/finetune_evaluate_r52.py) are scripts for fine-tuning training + model saving + inference on the WebKB dataset and Reuters 21578 dataset respectively. 
- Fine-tuning training and inference can begin by executing the script directly using python3 statements. 
- The data files corresponding to the WebKB dataset are [train.tsv](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/train.tsv) and [test.tsv](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/test.tsv), while the data files corresponding to Reuters 21578 are [r52train.tsv](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/r52train.tsv) and [r52test.tsv](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/r52test.tsv). 
- We further saved the model and weight file of each epoch during the training, and the script to implement this function is [train.py](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/train.py). After the train.py file is executed, the model and weight of each epoch can be loaded to inference the test set using [inference.py](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/inference.py). 
- On the basis of saving the weight of the model, we realized the interaction function in the [interaction.py](https://github.com/wygsjtu/finetune-Gluonnlp-BERT/blob/master/src/interaction.py) script. 
- Running the script will make inference prediction on an input paragraph of words or sentences, and output the label corresponding to the prediction result. The output information for running the finetune_evaluator.py and finetune_evaluater52.py scripts is recorded in the /loginfo folder.
## Datasets
[WebKB dataset](http://www.google.com/url?q=http%3A%2F%2Fwww.cs.cmu.edu%2Fafs%2Fcs.cmu.edu%2Fproject%2Ftheo-20%2Fwww%2Fdata%2F&sa=D&sntz=1&usg=AFQjCNEOrlUR_oci7gC1zHrEjGG7ujksqQ) and [Reuters 21578 dataset](http://www.google.com/url?q=http%3A%2F%2Fwww.daviddlewis.com%2Fresources%2Ftestcollections%2Freuters21578%2F&sa=D&sntz=1&usg=AFQjCNEaq3FcnH_SctlxbLcIWWehjWDpFA) can be downloaded [here](https://drive.google.com/drive/folders/1p3-IeJ1MMAdtjBEtOj3RMIvuYtaGkjpi?usp=sharing)
