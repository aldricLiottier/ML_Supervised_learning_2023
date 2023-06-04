# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cDVZBIP9l4lZKayLeMBlunvTJFKpDWh4
"""

import numpy as np

np.random.seed(42)

num_rows = 300
num_cols = 6

means = [10, 20, 30, 40, 50, 60]

stds = [1, 2, 3, 4, 5, 6]

dataset = np.empty((num_rows, num_cols))

for col in range(num_cols):
    if col == 3:
        column_data = np.random.randint(1, 100, size=num_rows)
    else:
        column_data = np.random.normal(means[col], stds[col], size=num_rows)
    dataset[:, col] = column_data

print(dataset)

print (np.mean(dataset, axis=0))

