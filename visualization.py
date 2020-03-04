import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('./data/student-por.csv', delimiter=";")

df['AVG_G'] = round((df['G1'] + df['G2'] + df['G3']) / 3)

print(df.iloc[:,6:].corr("pearson"))
