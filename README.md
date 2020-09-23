数据集下载：http://www.cs.cmu.edu/~glai1/data/cloth/

数据预处理
先运行:
python3 data_util.py
注：需要修改data_util.py中207-210行的相关路径

模型微调 
运行:
python main.py --output_dir './output' \
--data_dir \
'./data' \
--bert_model \
./uncased_base/bert-base-uncased.tar.gz \
--do_eval --do_train --train_batch_size 4 \
--output_dir EXP/ \
--learning_rate 1e-5 --num_train_epochs 4 

同样，需要修改预训练模型的本地路径，--bert_model参数也可以直接输入bert模型名，会自动下载对应模型，具体见main.py中的说明