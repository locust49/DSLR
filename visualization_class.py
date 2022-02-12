import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression_class import LogisticRegression

class DataVisualization:

	def __init__(	self,
					dataframe: pd.DataFrame,
					target_name: str)-> None:
		self.data = dataframe
		self.data_num = dataframe.select_dtypes(include=[np.number])
		self.target = dataframe[target_name]
		self.data_normed = LogisticRegression.normalize_data(self.data)
		self.column = self.data.select_dtypes(include=[np.number]).columns
		self.describe = self.target.describe()
		self.verbose = True
		# print(self.describe)
		# self.index_normed = self.data_normed.index.tolist()

	def histogram(self) -> None:
		MAX_ROW = 3
		MAX_COL = int(self.data.shape[1] / MAX_ROW) + 1
		fig, axes = plt.subplots(MAX_ROW, MAX_COL, figsize=(18,15))
		fig.suptitle('Histograms')
		row, col = 0, 0
		for index, subject in enumerate(self.column):
			if index % MAX_COL == 0 and index != 0:
				row += 1
				col = 0
			if row >= MAX_ROW:
				break
			sns.histplot(data=self.data, legend=True, ax=axes[row, col], x=subject, hue=self.target, palette=sns.color_palette('bright')[:4])
			col += 1

		legend = axes[row - 1, col - 1].get_legend()
		fig.legend(legend, loc='upper center')
		for r in range(row, MAX_ROW):
			for c in range(col, MAX_COL):
				fig.delaxes(axes[r, c])
		plt.show()

	@LogisticRegression.verbose_func
	def pairplot(self) -> None:
		graph = sns.pairplot(data=self.data, hue="Hogwarts House", diag_kind="kde", corner=True)
		# graph.map_lower(sns.kdeplot, levels=4, color=".4")