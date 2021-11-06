from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import h5py

def preprocess_data(data, n):
    '''
    data: n_sample, m_feature
    '''
    selector = VarianceThreshold()
    selector.fit(data)
    vars = selector.variances_
    vars = sorted(vars, reverse = True)
    thresh = vars[n]
    # Remove features whose variance is lower than the threshold value
    selector = VarianceThreshold(threshold=thresh)
    value = selector.fit_transform(data)
    features = selector.get_support(indices=True)
    data = pd.DataFrame(value, index=None, columns=None)
    # Perform maximum and minimum normalization
    data = (data - data.min())/(data.max() - data.min())
    return data


if __name__ == '__main__':
    file_path = r'GSE70256.csv'
    data = pd.read_csv(file_path, error_bad_lines = False)
    data.index = list(data.iloc[:, 0])
    data = data.drop(['Unnamed: 0'], axis=1)

    data_pre = preprocess_data(data, int(data.shape[1] * 0.20))
    data_pre.to_csv(r'gse70256/data/1.csv', header=True, index=True)
