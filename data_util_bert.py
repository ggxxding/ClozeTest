import os, sys
import glob
import time

import numpy as np
import torch
import json
import nltk
import argparse
import fnmatch
import random
import copy

from transformers.tokenization_bert import BertTokenizer

def get_json_file_list(data_dir):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, '*.json'):
            files.append(os.path.join(root, filename))
    return files

def tokenize_ops(ops, tokenizer):
    ret = []
    for i in range(4):
        ret.append(tokenizer.tokenize(ops[i]))
    return ret

def to_device(L, device):
    if (type(L) != list):
        return L.to(device)
    else:
        ret = []
        for item in L:
            ret.append(to_device(item, device))
        return ret

class ClothSample(object):
    def __init__(self):
        self.article = None
        self.ph = []
        self.ops = []
        self.ans = []
        self.high = 0
                    
    def convert_tokens_to_ids(self, tokenizer):
        #print(self.article)
        self.article = tokenizer.convert_tokens_to_ids(self.article)
        #print(self.article)
        self.article = torch.Tensor(self.article)

        for i in range(len(self.ops)):
            for k in range(4):
                self.ops[i][k] = tokenizer.convert_tokens_to_ids(self.ops[i][k])
                self.ops[i][k] = torch.Tensor(self.ops[i][k])
                #print(self.ops[i][k].size())
        self.ph = torch.Tensor(self.ph)
        self.ans = torch.Tensor(self.ans)



        
class Preprocessor(object):
    def __init__(self, args, device='cpu'):
        print(args.bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_dir = args.data_dir
        file_list = get_json_file_list(args.data_dir)
        self.data = []
        self.shortt=0
        self.longg=0
        #max_article_len = 0
        for file_name in file_list:
            data = json.loads(open(file_name, 'r').read())
            data['high'] = 0
            if ('high' in file_name):
                data['high'] = 1
            self.data.append(data)
            #max_article_len = max(max_article_len, len(nltk.word_tokenize(data['article'])))
        self.data_objs = []
        high_cnt = 0
        middle_cnt = 0

        for sample in self.data:
            high_cnt += sample['high']
            middle_cnt += (1 - sample['high'])
            self.data_objs += self._create_sample(sample)
            #print(self.data_objs[-1].ph)
            #break
        print('high school sample:', high_cnt)
        print('middle school sample:', middle_cnt)
        print('<512:',self.shortt)
        print('>512:',self.longg)
        for i in range(len(self.data_objs)):
            self.data_objs[i].convert_tokens_to_ids(self.tokenizer)
            #break

        torch.save(self.data_objs, args.save_name)
        
    
    def _create_sample(self, data):
        cnt = 0 
        #print(data['article'])
        #tokenize: string -> token list
        #convert_tokens_to_ids: token list -> id list
        article = self.tokenizer.tokenize(data['article'])
        #print(article)
        '''temp_len=0
        has_mask=0
        sample_list=[]
        for p in range(len(article)):
            if (article[p] == '_'):
                if (has_mask == 0):
                    temp_sample = ClothSample()
                    temp_sample.high = data['high']
                    has_mask = 1
                article[p] = '[MASK]'
                temp_sample.ph.append(p-temp_len)
                ops = tokenize_ops(data['options'][cnt], self.tokenizer)
                temp_sample.ops.append(ops)
                temp_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                cnt += 1
            elif (has_mask == 1 and (article[p] == '.' or article[p] == '?')):
                has_mask = 0
                temp_sample.article = article[temp_len:p+1]
                sample_list.append(temp_sample)
                #print(temp_sample.ans)
                temp_len=p+1
        return sample_list'''

        #origin
        if (len(article) <= 512):
            self.shortt+=1
            sample = ClothSample()
            sample.article = article
            sample.high = data['high']
            for p in range(len(article)):#p为词指针，ph为[MASK]指针列表
                if (sample.article[p] == '_'):
                    sample.article[p] = '[MASK]'
                    sample.ph.append(p)
                    ops = tokenize_ops(data['options'][cnt], self.tokenizer)
                    sample.ops.append(ops)
                    sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    cnt += 1
            return [sample]
        else:
            self.longg+=1
            first_sample = ClothSample()
            second_sample = ClothSample()
            first_sample.high = data['high']
            second_sample.high = data['high']
            second_s = len(article) - 512
            for p in range(len(article)):
                if (article[p] == '_'):
                    article[p] = '[MASK]'
                    ops = tokenize_ops(data['options'][cnt], self.tokenizer)
                    if (p < 512):
                        first_sample.ph.append(p)
                        first_sample.ops.append(ops)
                        first_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    else:
                        second_sample.ph.append(p - second_s)
                        second_sample.ops.append(ops)
                        second_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    cnt += 1
            first_sample.article = article[:512]
            second_sample.article = article[-512:]
            if (len(second_sample.ans) == 0):
                return [first_sample]
            else:
                return [first_sample, second_sample]

        #文章分为前后两部分填空
        '''if (len(article) <= 512):
            self.shortt+=1
            sample = ClothSample()
            sample.article = article
            sample.high = data['high']
            for p in range(len(article)):#p为词指针，ph为[MASK]指针列表
                if (sample.article[p] == '_'):
                    sample.article[p] = '[MASK]'
                    sample.ph.append(p)
                    ops = tokenize_ops(data['options'][cnt], self.tokenizer)
                    sample.ops.append(ops)
                    sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    cnt += 1

            #change [MASK] to answer
            sample_list = []
            temp_sample = ClothSample()
            temp_sample.article = copy.deepcopy(sample.article)
            temp_sample.high = sample.high
            temp_sample2 = ClothSample()
            temp_sample2.article = copy.deepcopy(sample.article)
            temp_sample2.high = sample.high
            for ques_ind in range(len(sample.ph)):
                if (ques_ind < len(sample.ph)/2):
                    temp_sample.ph.append(sample.ph[ques_ind])
                    temp_sample.ops.append(sample.ops[ques_ind])
                    temp_sample.ans.append(sample.ans[ques_ind])
                    temp_sample2.article[sample.ph[ques_ind]] = sample.ops[ques_ind][sample.ans[ques_ind]][0]
                if (ques_ind >= len(sample.ph)/2):
                    temp_sample2.ph.append(sample.ph[ques_ind])
                    temp_sample2.ops.append(sample.ops[ques_ind])
                    temp_sample2.ans.append(sample.ans[ques_ind])
                    temp_sample.article[sample.ph[ques_ind]] = sample.ops[ques_ind][sample.ans[ques_ind]][0]
            sample_list.append(temp_sample)
            sample_list.append(temp_sample2)
            return sample_list
        else:
            self.longg+=1
            first_sample = ClothSample()
            second_sample = ClothSample()
            first_sample.high = data['high']
            second_sample.high = data['high']
            second_s = len(article) - 512
            for p in range(len(article)):
                if (article[p] == '_'):
                    article[p] = '[MASK]'
                    ops = tokenize_ops(data['options'][cnt], self.tokenizer)
                    if (p < 512):
                        first_sample.ph.append(p)
                        first_sample.ops.append(ops)
                        first_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    else:
                        second_sample.ph.append(p - second_s)
                        second_sample.ops.append(ops)
                        second_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    cnt += 1
            first_sample.article = article[:512]
            second_sample.article = article[-512:]

            sample_list = []
            temp_first_sample = ClothSample()
            temp_first_sample.article = copy.deepcopy(first_sample.article)
            temp_first_sample.high = first_sample.high
            temp_first_sample2 = ClothSample()
            temp_first_sample2.article = copy.deepcopy(first_sample.article)
            temp_first_sample2.high = first_sample.high
            temp_second_sample = ClothSample()
            temp_second_sample.article = copy.deepcopy(second_sample.article)
            temp_second_sample.high = second_sample.high
            temp_second_sample2 = ClothSample()
            temp_second_sample2.article = copy.deepcopy(second_sample.article)
            temp_second_sample2.high = second_sample.high
            for ques_ind in range(len(first_sample.ph)):
                if (ques_ind < len(first_sample.ph)/2):
                    temp_first_sample.ph.append(first_sample.ph[ques_ind])
                    temp_first_sample.ops.append(first_sample.ops[ques_ind])
                    temp_first_sample.ans.append(first_sample.ans[ques_ind])
                    temp_first_sample2.article[first_sample.ph[ques_ind]] = first_sample.ops[ques_ind][first_sample.ans[ques_ind]][0]
                if (ques_ind >= len(first_sample.ph)/2):
                    temp_first_sample2.ph.append(first_sample.ph[ques_ind])
                    temp_first_sample2.ops.append(first_sample.ops[ques_ind])
                    temp_first_sample2.ans.append(first_sample.ans[ques_ind])
                    temp_first_sample.article[first_sample.ph[ques_ind]] = first_sample.ops[ques_ind][first_sample.ans[ques_ind]][0]
            sample_list.append(temp_first_sample)
            sample_list.append(temp_first_sample2)
            if (len(second_sample.ans) == 0):
                return sample_list
            else:
                for ques_ind in range(len(second_sample.ans)):
                    if (ques_ind >= len(second_sample.ph)/2):
                        temp_second_sample2.ph.append(second_sample.ph[ques_ind])
                        temp_second_sample2.ops.append(second_sample.ops[ques_ind])
                        temp_second_sample2.ans.append(second_sample.ans[ques_ind])
                        temp_second_sample.article[second_sample.ph[ques_ind]] = second_sample.ops[ques_ind][second_sample.ans[ques_ind]][0]
                    if (ques_ind < len(second_sample.ph)/2):
                        temp_second_sample.ph.append(second_sample.ph[ques_ind])
                        temp_second_sample.ops.append(second_sample.ops[ques_ind])
                        temp_second_sample.ans.append(second_sample.ans[ques_ind])
                        temp_second_sample2.article[second_sample.ph[ques_ind]] = second_sample.ops[ques_ind][second_sample.ans[ques_ind]][0]
                sample_list.append(temp_second_sample)
                #print(temp_second_sample2.article)
                #print(temp_second_sample2.ops)
                return sample_list

            if (len(second_sample.ans) == 0):
                return [first_sample]
            else:
                return [first_sample, second_sample]'''

        #部分单词答案由多个token组成，插入时把他们全部插入，会使文章变长，超过512会报错，因此要先进行插入再分段
        '''if (len(article) <= 512):
            self.shortt+=1
            sample = ClothSample()
            sample.article = article
            sample.high = data['high']

            for p in range(len(article)):#p为词指针，ph为[MASK]指针列表
                if (sample.article[p] == '_'):
                    sample.article[p] = '[MASK]'
                    sample.ph.append(p)
                    ops = tokenize_ops(data['options'][cnt], self.tokenizer)
                    sample.ops.append(ops)
                    sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    cnt += 1

            #change [MASK] to answer
            sample_list = []
            temp_sample = ClothSample()
            temp_sample.article = copy.deepcopy(sample.article)
            temp_sample.high = sample.high
            temp_sample2 = ClothSample()
            temp_sample2.article = copy.deepcopy(sample.article)
            temp_sample2.high = sample.high
            temp_len = 0
            temp_len2 = 0
            for ques_ind in range(len(sample.ph)):
                if (ques_ind %2 ==0):

                    temp_sample.ph.append(sample.ph[ques_ind]+temp_len)
                    temp_sample.ops.append(sample.ops[ques_ind])
                    temp_sample.ans.append(sample.ans[ques_ind])
                    temp_sample2.article[sample.ph[ques_ind]+temp_len2] = sample.ops[ques_ind][sample.ans[ques_ind]][0]
                    if(len(sample.ops[ques_ind][sample.ans[ques_ind]])>1):
                        for ind in range(1,len(sample.ops[ques_ind][sample.ans[ques_ind]])):
                            temp_len2 += 1
                            temp_sample2.article.insert(sample.ph[ques_ind]+temp_len2,sample.ops[ques_ind][sample.ans[ques_ind]][ind])
                if (ques_ind %2 ==1):
                    temp_sample2.ph.append(sample.ph[ques_ind]+temp_len2)
                    temp_sample2.ops.append(sample.ops[ques_ind])
                    temp_sample2.ans.append(sample.ans[ques_ind])
                    temp_sample.article[sample.ph[ques_ind]+temp_len] = sample.ops[ques_ind][sample.ans[ques_ind]][0]
                    if(len(sample.ops[ques_ind][sample.ans[ques_ind]])>1):
                        for ind in range(1,len(sample.ops[ques_ind][sample.ans[ques_ind]])):
                            temp_len += 1
                            temp_sample.article.insert(sample.ph[ques_ind]+temp_len,sample.ops[ques_ind][sample.ans[ques_ind]][ind])
            sample_list.append(temp_sample)
            sample_list.append(temp_sample2)
            return sample_list
        else:
            self.longg+=1
            first_sample = ClothSample()
            second_sample = ClothSample()
            first_sample.high = data['high']
            second_sample.high = data['high']
            second_s = len(article) - 512
            for p in range(len(article)):
                if (article[p] == '_'):
                    article[p] = '[MASK]'
                    ops = tokenize_ops(data['options'][cnt], self.tokenizer)
                    if (p < 512):
                        first_sample.ph.append(p)
                        first_sample.ops.append(ops)
                        first_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    else:
                        second_sample.ph.append(p - second_s)
                        second_sample.ops.append(ops)
                        second_sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    cnt += 1
            first_sample.article = article[:512]
            second_sample.article = article[-512:]

            sample_list = []
            temp_first_sample = ClothSample()
            temp_first_sample.article = copy.deepcopy(first_sample.article)
            temp_first_sample.high = first_sample.high
            temp_first_sample2 = ClothSample()
            temp_first_sample2.article = copy.deepcopy(first_sample.article)
            temp_first_sample2.high = first_sample.high
            temp_second_sample = ClothSample()
            temp_second_sample.article = copy.deepcopy(second_sample.article)
            temp_second_sample.high = second_sample.high
            temp_second_sample2 = ClothSample()
            temp_second_sample2.article = copy.deepcopy(second_sample.article)
            temp_second_sample2.high = second_sample.high
            temp_first_len = 0
            temp_first_len2 = 0
            temp_second_len = 0
            temp_second_len2 = 0
            for ques_ind in range(len(first_sample.ph)):
                if (ques_ind %2 ==0):
                    temp_first_sample.ph.append(first_sample.ph[ques_ind]+temp_first_len)
                    temp_first_sample.ops.append(first_sample.ops[ques_ind])
                    temp_first_sample.ans.append(first_sample.ans[ques_ind])
                    temp_first_sample2.article[first_sample.ph[ques_ind]+temp_first_len2] = first_sample.ops[ques_ind][first_sample.ans[ques_ind]][0]
                    if(len(first_sample.ops[ques_ind][first_sample.ans[ques_ind]])>1):
                        for ind in range(1,len(first_sample.ops[ques_ind][first_sample.ans[ques_ind]])):
                            temp_first_len2 += 1
                            temp_first_sample2.article.insert(first_sample.ph[ques_ind]+temp_first_len2,first_sample.ops[ques_ind][first_sample.ans[ques_ind]][ind])
                if (ques_ind %2 ==1):
                    temp_first_sample2.ph.append(first_sample.ph[ques_ind]+temp_first_len2)
                    temp_first_sample2.ops.append(first_sample.ops[ques_ind])
                    temp_first_sample2.ans.append(first_sample.ans[ques_ind])
                    temp_first_sample.article[first_sample.ph[ques_ind]+temp_first_len] = first_sample.ops[ques_ind][first_sample.ans[ques_ind]][0]
                    if(len(first_sample.ops[ques_ind][first_sample.ans[ques_ind]])>1):
                        for ind in range(1,len(first_sample.ops[ques_ind][first_sample.ans[ques_ind]])):
                            temp_first_len += 1
                            temp_first_sample.article.insert(first_sample.ph[ques_ind]+temp_first_len,first_sample.ops[ques_ind][first_sample.ans[ques_ind]][ind])
            sample_list.append(temp_first_sample)
            sample_list.append(temp_first_sample2)
            if (len(second_sample.ans) == 0):
                return sample_list
            else:
                for ques_ind in range(len(second_sample.ans)):
                    if (ques_ind %2 == 0):
                        temp_second_sample.ph.append(second_sample.ph[ques_ind]+temp_second_len)
                        temp_second_sample.ops.append(second_sample.ops[ques_ind])
                        temp_second_sample.ans.append(second_sample.ans[ques_ind])
                        temp_second_sample2.article[second_sample.ph[ques_ind]+temp_second_len2] = second_sample.ops[ques_ind][second_sample.ans[ques_ind]][0]
                        if(len(second_sample.ops[ques_ind][second_sample.ans[ques_ind]])>1):
                            for ind in range(1,len(second_sample.ops[ques_ind][second_sample.ans[ques_ind]])):
                                temp_second_len2 += 1
                                temp_second_sample2.article.insert(second_sample.ph[ques_ind]+temp_second_len2,second_sample.ops[ques_ind][second_sample.ans[ques_ind]][ind])
                    if (ques_ind %2 == 1):
                        temp_second_sample2.ph.append(second_sample.ph[ques_ind]+temp_second_len2)
                        temp_second_sample2.ops.append(second_sample.ops[ques_ind])
                        temp_second_sample2.ans.append(second_sample.ans[ques_ind])
                        temp_second_sample.article[second_sample.ph[ques_ind]+temp_second_len] = second_sample.ops[ques_ind][second_sample.ans[ques_ind]][0]
                        if(len(second_sample.ops[ques_ind][second_sample.ans[ques_ind]])>1):
                            for ind in range(1,len(second_sample.ops[ques_ind][second_sample.ans[ques_ind]])):
                                temp_second_len += 1
                                print(temp_second_sample.article[second_sample.ph[ques_ind]])
                                temp_second_sample.article.insert(second_sample.ph[ques_ind]+temp_second_len,second_sample.ops[ques_ind][second_sample.ans[ques_ind]][ind])
                sample_list.append(temp_second_sample)
                if(len(temp_second_sample2.ph)>0):
                    sample_list.append(temp_second_sample2)
                return sample_list

            if (len(second_sample.ans) == 0):
                return [first_sample]
            else:
                return [first_sample, second_sample]'''

class Loader(object):
    def __init__(self, data_dir, data_file, cache_size, batch_size, device='cpu'):
        #self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_dir = os.path.join(data_dir, data_file)
        print('loading {}'.format(self.data_dir))
        self.data = torch.load(self.data_dir)
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.data_num = len(self.data)
        self.device = device
    
    def _batchify(self, data_set, data_batch):
        max_article_length = 0
        max_option_length = 0
        max_ops_num = 0
        bsz = len(data_batch)
        for idx in data_batch:
            data = data_set[idx]
            max_article_length = max(max_article_length, data.article.size(0))
            for ops in data.ops:
                for op in ops:
                    max_option_length = max(max_option_length, op.size(0))
            max_ops_num  = max(max_ops_num, len(data.ops))
        articles = torch.zeros(bsz, max_article_length).long()
        articles_mask = torch.ones(articles.size())
        options = torch.zeros(bsz, max_ops_num, 4, max_option_length).long()
        options_mask = torch.ones(options.size())
        answers = torch.zeros(bsz, max_ops_num).long()
        mask = torch.zeros(answers.size())
        question_pos = torch.zeros(answers.size()).long()
        high_mask = torch.zeros(bsz) #indicate the sample belong to high school set
        for i, idx in enumerate(data_batch):
            data = data_set[idx]
            articles[i, :data.article.size(0)] = data.article
            articles_mask[i, data.article.size(0):] = 0
            for q, ops in enumerate(data.ops):
                for k, op in enumerate(ops):
                    options[i,q,k,:op.size(0)] = op
                    options_mask[i,q,k, op.size(0):] = 0
            for q, ans in enumerate(data.ans):
                answers[i,q] = ans
                mask[i,q] = 1
            for q, pos in enumerate(data.ph):
                question_pos[i,q] = pos
            high_mask[i] = data.high
        inp = [articles, articles_mask, options, options_mask, question_pos, mask, high_mask]
        tgt = answers
        return inp, tgt
                
                
    def data_iter(self, shuffle=True):
        if (shuffle == True):
            random.shuffle(self.data)
        seqlen = torch.zeros(self.data_num)
        for i in range(self.data_num):
            seqlen[i] = self.data[i].article.size(0)
        cache_start = 0
        while (cache_start < self.data_num):
            cache_end = min(cache_start + self.cache_size, self.data_num)
            cache_data = self.data[cache_start:cache_end]
            seql = seqlen[cache_start:cache_end]
            _, indices = torch.sort(seql, descending=True)
            batch_start = 0
            while (batch_start + cache_start < cache_end):
                batch_end = min(batch_start + self.batch_size, cache_end - cache_start)
                data_batch = indices[batch_start:batch_end]
                inp, tgt = self._batchify(cache_data, data_batch)
                inp = to_device(inp, self.device)
                tgt = to_device(tgt, self.device)
                yield inp, tgt
                batch_start += self.batch_size
            cache_start += self.cache_size
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model",
                        default='./data',
                        type=str,
                        required=True,
                        help=" ")
    args = parser.parse_args()
    #'''
    data_collections = ['train', 'valid', 'test']
    for item in data_collections:    
        args.data_dir = './CLOTH/{}'.format(item)
        args.pre = args.post = 0
        args.bert_model =args.bert_model
        args.save_name = './data/{}-'.format(item)+args.bert_model+'.pt'
        data = Preprocessor(args)
    '''
    args.data_dir = './data/'
    args.bert_model = 'bert-large-uncased'
    args.cache_size = 32
    args.batch_size = 2
    train_data = Loader(args.data_dir, 'valid.pt', args.cache_size, args.batch_size)
    cnt = 0
    for inp, tgt in train_data.data_iter():
        articles, articles_mask, options, options_mask, question_pos = inp
    '''
