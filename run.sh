python main.py --output_dir './output' \
--data_dir \
'./data' \
--bert_model \
./uncased_base/bert-base-uncased.tar.gz \
--do_eval --do_train --train_batch_size 4 \
--output_dir EXP/ \
--learning_rate 1e-5 --num_train_epochs 4 