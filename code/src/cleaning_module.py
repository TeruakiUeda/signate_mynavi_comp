import numpy as np
import pandas as pd
import re


def splitting_word(df_input, column, split_word = '', n=2):
    """ 使えるか怪しい
    """
    df_out = df_input[column].str.split(split_word, n, expand=True)
    df_out.columns = [f'{column}_{i}' for i in range(n)]
    return df_out

def pickup_num_feature(df_input, columns):
    df_out = pd.DataFrame()
    for c in columns:
        df_out[f'{c}_'] = [re.sub('[^0-9]', "", i) for i in df_input[c]]
    return df_out


"""()で囲われた部分を取り除く"""
re.sub("\(.*?\)", "", text)