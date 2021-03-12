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

data_dir = './data'


def get_json_file_list(data_dir):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, '*.json'):
            files.append(os.path.join(root, filename))
    return files

def main():
	file_list = get_json_file_list(data_dir)
	char2num={'A':0,'B':1,'C':2,'D':3}
	high=0
	middle=0

	wrong_avg_count=0
	wrong_avg_counth=0
	wrong_avg_countm=0

	wrong_count=0
	right_count=0
	wrong_counth=0
	wrong_countm=0
	right_counth=0
	right_countm=0

	wrongAvg=0
	numAvg=0
	wrongAvgh=0
	numAvgh=0
	wrongAvgm=0
	numAvgm=0
	wrongOur=0
	numOur=0
	wrongOurh=0
	numOurh=0
	wrongOurm=0
	numOurm=0
	for file_name in file_list:
		data = json.loads(open(file_name, 'r').read())
		if(data['high'][0]==1):
			high+=1
		else:
			middle+=1
		#['article', 'falseList', 'option', 'high']
		if 'Avg' in file_name:
			#print(file_name)
			#print(data['falseList'])
			#print(len(data['falseList']))
			#print(data['falseList'])
			wrong_avg_count+=len(data['falseList'])
			if(data['high'][0]==1):
				wrong_avg_counth+=len(data['falseList'])
			else:
				wrong_avg_countm+=len(data['falseList'])
			flag1=0
			flag2=0
			for ind in data['falseList']:
				#print(data['option'][ind][-1])
				#print(len(data['option']))
				if(data['option'][ind][char2num[data['option'][ind][-2]]].count('[PAD]')!=data['option'][ind][char2num[data['option'][ind][-1]]].count('[PAD]')):
					#print(data['option'][ind][char2num[data['option'][ind][-2]]].count('[PAD]'))
					#print(data['option'][ind][char2num[data['option'][ind][-1]]].count('[PAD]'))
					flag1=1
					#print(data['article'][:50],1)
					if(data['high'][0]==1):
						wrong_counth+=1
					else:
						wrong_countm+=1
					wrong_count += 1
					#二次循环,查找与avg的错题相同的our错题
					for file_name2 in file_list:
						data2 = json.loads(open(file_name2, 'r').read())
						if ('Avg' not in file_name2) and (data['article']==data2['article']):
							#print(data2['article'][:50],2)
							if flag2 == 0:
								print(data2['article'][:25])
								flag2 = 1#防止重复计算
								wrongOur+=len(data2['falseList'])
								numOur+=len(data2['option'])
								if data2['high'][0] == 1:
									wrongOurh+=len(data2['falseList'])
									numOurh+=len(data2['option'])
								else:
									wrongOurm+=len(data2['falseList'])
									numOurm+=len(data2['option'])
							if (data2['option'][ind][-1]==data2['option'][ind][-2]):
								if(data2['high'][0]==1):
									right_counth+=1
								else:
									right_countm+=1
								right_count += 1
								#print('found')
								#print(data2['option'][ind])
			if flag1 == 1:
				print(data['article'][:25])
				if data['high'][0] == 1:
					wrongAvgh+=len(data['falseList'])
					numAvgh+=len(data['option'])
				else:
					wrongAvgm+=len(data['falseList'])
					numAvgm+=len(data['option'])
				wrongAvg+=len(data['falseList'])
				numAvg+=len(data['option'])


	print('high: ',high)
	print('middle: ',middle)
	print(right_count)
	print(wrong_count)
	print(right_counth)
	print(wrong_counth)
	print(right_countm)
	print(wrong_countm)
	print('question_num_TL:', numAvg)
	print('wrong_TL:', wrongAvg)
	print('Avg:', 1-wrongAvg/numAvg)
	print('Our:', 1-wrongOur/numOur)
	print('AvgH:',1-wrongAvgh/numAvgh)
	print('OurH:',1-wrongOurh/numOurh)
	print('AvgM:',1-wrongAvgm/numAvgm)
	print('OurM:',1-wrongOurm/numOurm)
	print('all wrong:',wrong_avg_count)
	print('middle wrong:',wrong_avg_countm)
	print('high wrong:',wrong_avg_counth)





if __name__ == "__main__":
    main()