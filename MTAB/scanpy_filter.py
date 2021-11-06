import numpy as np
import pandas as pd
import scanpy as sc
import h5py
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, facecolor='white')

results_file = '/home/Ganyanglan/HXY/zinb_sdnc_local/MAGIC_DSC/MTAB/data/mtab1.h5'
data_mat = h5py.File(results_file, "r+")
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])

adata = sc.AnnData(x)
# 添加序列号（index）
adata.var_names_make_unique()
print(adata)
# 显示在所有细胞中每个单个细胞中产生最高计数值的那些基因，这里选的是前20的
sc.pl.highest_expr_genes(adata, n_top=20, )
# 筛选出在不到3个细胞中检测到的19024个基因
# 不明白
sc.pp.filter_cells(adata, min_genes=200)
# 过滤掉低于3个细胞内表达的基因，比如2000个基因，在绝大多数（n-3）个细胞里都不表达
sc.pp.filter_genes(adata, min_cells=3)
# 将线粒体基因组注释为“ mt”
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
#计算出的质量测度的小提琴图：
# 计数矩阵中表达的基因数：n_genes_by_counts
# 每个单元格的总计数：total_counts
# 线粒体基因计数的百分比：pct_counts_mt
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
# 删除表达太多线粒体基因或总数过多的细胞：
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# 实际上是通过切片AnnData对象来进行过滤
# adata = adata[adata.obs.n_genes_by_counts < 2500, :]
# adata = adata[adata.obs.pct_counts_mt < 5, :]

# 总计数归一化（库大小正确）数据矩阵 X 每个单元最多10,000次读取，因此每个单元之间的计数变得可比
sc.pp.normalize_total(adata, target_sum=1e4)
# 对数据进行对数
sc.pp.log1p(adata)
# 鉴定高度可变的基因
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# 提取高度可变的基因
#     完成（0:00:00）
# ->添加
#     'highly_variable'，布尔向量（adata.var）
#     '均值'，浮点向量（adata.var）
#     '分散'，浮点向量（adata.var）
#     'dispersions_norm'，浮点向量（adata.var）
sc.pl.highly_variable_genes(adata)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
print(adata.X)
# sc.pp.regress_out(adata)
# 将每个基因缩放至单位方差。剪辑值超过标准偏差10。
# 这一句会让基因表达值为负值
# sc.pp.scale(adata, max_value=10)

adata.write_h5ad(r'/home/Ganyanglan/HXY/zinb_sdnc_local/MAGIC_DSC/MTAB/data/mtab_new.h5ad')
# X = adata.X
# X_raw = adata.raw.X
# print(X)
# x = np.array(X)
# x.to_csv(r'data/mouse_new.csv', header=True, index=True)