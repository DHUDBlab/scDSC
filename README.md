# scDSC
We propose a new method, named scDSC, to integrate the structural information into deep clustering of scRNA-seq data. The proposed scDSC consists of a ZINB-based autoencoder, a graph neural network (GNN) module, and a mutual-supervised module. To learn the data representation from the sparse and zero-inflated scRNA-seq data, we add a ZINB model to the basic autoencoder. The GNN module is introduced to capture the structural information among cells. By joining the ZINB-based autoencoder with the GNN module, the model transfers the data representation learned by autoencoder to the corresponding GNN layer.

Furthermore, we adopt a mutual supervised strategy to unify these two different deep neural architectures and to guide the clustering task. Extensive experimental results on six real scRNA-seq datasets demonstrate that scDSC outperforms state-of-the-art methods in terms of clustering accuracy and scalability.

1. Data preprocessing. After we obtain the scRNA-seq data, we need to make preliminary processing of the gene expression data. After the data is preprocessed, it is stored in the data directory. (Use scanpy_filter.py to preprocess the data). Due to the limitation of github space, we cannot put the complete data on it. Please download the detailed data file at the following website.
2. Generate graphs. We execute the calcu_graph_XX.py file, generate the graph required for input, and store it in the graph folder.
3. Pre-training. In order to obtain better training results, we conducted pre-training. By executing pretrain_XX.py, a pre-training model is generated. pkl file, store it in the model folder.
4. Training. Run the dsc_XX.py file to train the final model.
5. Remarks: The specific data file can be downloaded from the following website:

| ***\*MTAB\****              | ***\*EMBL-EBOI\****     | ***\*https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-3929/?query=1529+**** |
| --------------------------- | ----------------------- | ------------------------------------------------------------ |
| ***\*lps\****               | ***\*GEO-NCBI\****      | ***\*https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17721**** |
| ***\*GSE70256\****          | ***\*GEO-NCBI\****      | ***\*https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi****     |
| ***\*PBMC\****              | ***\*10X\****           | ***\*https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k**** |
| ***\*Mouse\****             | ***\*Microwell-seq\**** | ***\*https://figshare.com/s/865e694ad06d5857db4b****        |
| ***\*Worm neuron cells\**** | ***\*sci-RNA-seq\****   | **[http://atlas.gs.washington.edu/worm-rna/docs**            |



