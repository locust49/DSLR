#!/usr/bin/python3
import matplotlib.pyplot as plt
from data_operations import *
from histogram import draw_histogram
from scatter_plot import draw_scatter_plot
import sys

merge_histogram = True
explain = False
data_source = './datasets/dataset_train.csv'
colors = ['red', 'lime', 'aqua', 'brown']
font_size = 0
text_font_size = 7
alpha=0.5

def explain_process():
	data_ops = data_operations(data_source=data_source)
	index = 1
	plt.rcParams.update({'font.size': font_size})
	plt.figure(figsize=(15, 8))
	plt.subplots_adjust(left=0.02,
                    bottom=0.05, 
                    right=0.99, 
                    top=0.97,
                    wspace=0, 
                    hspace=0)
	for i in range(len(data_ops.classes)):
		for j in range(len(data_ops.classes)):
			subject1 = data_ops.classes[i]
			subject2 = data_ops.classes[j]
			plt.subplot(len(data_ops.classes), len(data_ops.classes), index)
			plt.xticks([])
			plt.yticks([])
			if i == 0:
				plt.gcf().text(0.99 / len(data_ops.classes) * j + 0.04, 0.98, "%15.15s" % (data_ops.classes[j]), fontsize=text_font_size, color="white", horizontalalignment="center")
			if j == 0:
				plt.gcf().text(0.01, 0.93 / len(data_ops.classes) * i + 0.06, "%7.7s" % (data_ops.classes[i]), fontsize=text_font_size, color="white", rotation=90, horizontalalignment="center")
			if i == j:
				draw_histogram(data_ops, subject1, fontsize=5)
			else:
				draw_scatter_plot(data_ops, subject1, subject2, fontsize=5)
			plt.xlabel("")
			plt.ylabel("")
			index+=1
	plt.gcf().legend(data_ops.houses, loc='lower right', fontsize=text_font_size*1.5, mode="expand", ncol=4)
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
	explain_process()