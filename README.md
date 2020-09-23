### 关于本项目
本项目基于Bert模型，使用CMU的CLOTH数据集进行微调和测试,最终在large的数据集下达到86%的准确率

数据预处理
先运行:

python3 data_util.py
注意：需要修改data_util.py中语料库的本地路径

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

同样，需要修改预训练模型的本地路径
