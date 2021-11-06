from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import h5py
#按方差选取特征，选取前n方差的特征，并归一化
def preprocess_data(data, n):
    '''
    data: n_sample, m_feature
    '''
    # 计算特征方差
    selector = VarianceThreshold()
    selector.fit(data)
    vars = selector.variances_
    # 对特征方差进行从大到小排序
    vars = sorted(vars, reverse = True)
    # 得到第n个方差的值
    thresh = vars[n]
    # 去除方差低于thresh值的特征
    selector = VarianceThreshold(threshold=thresh)
    value = selector.fit_transform(data)
    features = selector.get_support(indices=True)  # 获得选取特征的列号
    data = pd.DataFrame(value, index=None, columns=None)
    # 进行最大最小值归一化
    data = (data - data.min())/(data.max() - data.min())
    return data


if __name__ == '__main__':
    # data_mat = h5py.File("data/gse70256_ori.h5", "r+")
    # data = np.array(data_mat['X'])
    # print(data)
    # print("dd一号")
    # data_pre = preprocess_data(data, int(data.shape[1] * 0.20))
    # data_pre.to_csv(r'gse70256/data/1.csv', header=True, index=True)

    file_path = r'data/gse70256_ori2.csv'
    data = pd.read_csv(file_path, error_bad_lines=False)
    print(data.shape)
    print("读取完成")
    print("dd一号")
    data_pre = preprocess_data(data, int(data.shape[1] * 0.20))
    data_pre.to_csv(r'data/gse70256_ori_filter1.csv', header=True, index=True)
