#!/usr/bin/python3
import matplotlib.pyplot as plt
from data_operations import *
import math
import sys
import re

merge_histogram = True
explain = False
colors = ['red', 'lime', 'aqua', 'brown']
font_size = 10
labels_regex = re.compile("<AxesSubplot:xlabel='(.*?)', *ylabel='(.*?)'")
annotation = None

def draw_scatter_plot(data_ops, subject1, subject2, fontsize=0):
	global colors

	data = data_ops.get_data().set_index('Hogwarts House')
	for i in range(len(data_ops.houses)):
		df = data.loc[data_ops.houses[i]]
		plt.scatter(df[subject1], df[subject2], s=1, color=colors[i], alpha=0.5)
	plt.xlabel(subject1, fontsize=fontsize)
	plt.ylabel(subject2, fontsize=fontsize)

def calculate_best_square(num):
	best_square = math.sqrt(num)
	return ((math.ceil(best_square), math.ceil(best_square)))

def update_annot(text):
	global annotation

	if not annotation:
		return
	annotation.set_text(text[0] + " x " + text[1])

def hover(event):
	try:
		res = re.findall(labels_regex, str(event.__dict__))
		update_annot(res[0])
	except Exception:
		pass

def explain_process():
	global annotation
	global font_size

	font_size = 0
	plt.rcParams.update({'font.size': font_size})
	data_ops = data_operations()
	index = 1
	fig = plt.figure(figsize=(15, 8))
	annotation = plt.annotate("Hover", (5, 5));
	fig.canvas.mpl_connect("motion_notify_event", hover)
	plt.subplots_adjust(left=0,
                    bottom=0,
                    right=1,
                    top=1,
                    wspace=0,
                    hspace=0)
	for i in range(len(data_ops.classes)):
		for j in range(i+1, len(data_ops.classes)):
			subject1 = data_ops.classes[i]
			subject2 = data_ops.classes[j]
			plot_size = calculate_best_square(((len(data_ops.classes) ** 2) // 2) - len(data_ops.classes) + 2)
			plt.subplot(plot_size[0], plot_size[1], index)
			draw_scatter_plot(data_ops, subject1, subject2)
			index+=1
	annotation = plt.gcf().text(0.68, 0.05, 'Course Score Correlation', fontsize=14)
	plt.gcf().legend(data_ops.houses, loc='lower right', fontsize=9)
	while True:
		if not plt.fignum_exists(1):
			exit(0)
		plt.pause(1)

def	show_result():
	data_ops = data_operations()
	draw_scatter_plot(data_ops, 'Astronomy', 'Defense Against the Dark Arts', fontsize=font_size)
	plt.gcf().legend(data_ops.houses, loc='upper right')
	plt.show()

def manage_arguments(args):
	global explain
	global merge_histogram

	if (len(args) != 1):
		for arg in args:
			if arg == "--explain":
				explain = True
			elif arg == "--no-merge":
				merge_histogram = False

if __name__ == '__main__':
	plt.style.use('dark_background')
	manage_arguments(sys.argv)
	if explain:
		explain_process()
	else:
		show_result()