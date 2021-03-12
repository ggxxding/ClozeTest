数据集下载：http://www.cs.cmu.edu/~glai1/data/cloth/

数据预处理
先运行:
python3 data_util(_bert).py
注：需要修改data_util.py中207-210行的相关路径

模型微调 
运行:
python main(_bert).py --output_dir './output' \
--data_dir \
'./data' \
--bert_model \
'bert-base-uncased' \
--do_eval --do_train --train_batch_size 4 \
--output_dir ./output \
--learning_rate 1e-5 --num_train_epochs 4 


同样，需要修改预训练模型的本地路径，--bert_model参数也可以直接输入bert模型名，会自动下载对应模型，具体见main.py中的说明

在214服务器上已经部署完毕，直接运行微调的代码即可

2020/10/15更新：
用albert、electra、roberta等模型进行cloth实验（结果都没有bert好），运行对应的data_util_model和main_model文件即可，相关参数均有说明。

2021/2/3:

analyse.py分析avg模型因为token length导致的错题和其中因为我们的改进而纠正的比率