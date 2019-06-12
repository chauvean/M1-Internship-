import pandas as pd
import numpy as np
import xlrd
from random import randint as rdi
import matplotlib.pyplot as plt

"""
dataframe with m values
df.rolling(n, k (default : n)) : if k correct values before ith sample, calculation will be made. 
"""


def Bolinger_Bands(prec_info, window_size, num_of_std):
    rolling_mean = prec_info.rolling(window=window_size).mean()
    rolling_std = prec_info.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)

    return rolling_mean, upper_band, lower_band

# encoding_dict = dict()
# df = pd.read_excel("bio_precede_full3.xlsx")
# for val, encoding_index in enumerate(df['Source']):
#     if val not in encoding_dict:
#         encoding_dict[val]= encoding_dict
# print(encoding_dict)
# print(encoding_dict)
# print(df['Source'].size)
# print(df)
# x = range(1000)
# prec_info = pd.Series([rdi(1, 100) for _ in range(1000)])
# a, b, c = Bolinger_Bands(prec_info, 20, 1)
# plt.plot(x, a)
# plt.show()
print("test passed")
