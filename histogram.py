#!/usr/bin/python3
import matplotlib.pyplot as plt
from data_operations import *
import math
import numpy as np
import sys

merge_histogram = True
data_source = './datasets/dataset_train.csv'
explain = False
colors = ['red', 'lime', 'aqua', 'brown']

def draw_histogram(data_ops, subject, fontsize=10):
	data = data_ops.get_data()
	df = data[['Hogwarts House', subject]].set_index('Hogwarts House')
	plt_data = []
	i = 0
	for house in data_ops.houses:
		if merge_histogram:
			plt.hist(np.concatenate(df.loc[house].to_numpy()), color=colors[i], alpha=0.5)
		else:
			plt_data.append(np.concatenate(df.loc[house].to_numpy()))
		i+=1
	if not merge_histogram:
		plt.hist(plt_data)
	plt.title(subject)
	plt.xlabel('Score', fontsize=fontsize)
	plt.ylabel('Frequency', fontsize=fontsize)

def explain_process():
	data_ops = data_operations(data_source=data_source)
	index = 1
	plt.figure(figsize=(15, 8))
	plt.subplots_adjust(left=0.05,
                    bottom=0.07, 
                    right=0.98, 
                    top=0.95,
                    wspace=0.6, 
                    hspace=0.6)
	for subject in data_ops.classes:
		plt.subplot(3, math.ceil(len(data_ops.classes)/3), index)
		draw_histogram(data_ops, subject)
		index+=1
	plt.gcf().legend(data_ops.houses, loc='lower right')
	plt.gcf().text(0.68, 0.18, 'Student Score Distribution By House', fontsize=14)
	plt.show()

def	show_result():
	data_ops = data_operations(data_source=data_source)
	draw_histogram(data_ops, 'Care of Magical Creatures')
	plt.gcf().legend(data_ops.houses, loc='upper right')
	plt.show()

def manage_arguments(args):
	global explain
	global merge_histogram
	global data_source

	if (len(args) != 1):
		for arg in args:
			if arg == "--explain":
				explain = True
			elif arg == "--no-merge":
				merge_histogram = False
			elif arg.startswith("--source="):
				data_source = arg[9:]

if __name__ == '__main__':
	plt.style.use('dark_background')
	manage_arguments(sys.argv)
	if explain:
		explain_process()
	else:
		show_result()