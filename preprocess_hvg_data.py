# @title Concatenate the data and compute HVGs (memory intensive for a Colab).
import scanpy as sc
import anndata as ad
import numpy as np
from scipy import sparse

# Load filtered files
files = [
    'data/GSE264667_hepg2_raw_singlecell_01_filtered.h5ad',
    'data/GSE264667_jurkat_raw_singlecell_01_filtered.h5ad',
    'data/K562_essential_normalized_singlecell_01_filtered.h5ad',
    'data/rpe1_normalized_singlecell_01_filtered.h5ad'
]

adatas = [ad.read_h5ad(f) for f in files]

# Concatenate with inner join on columns
x = ad.concat(adatas, join='inner')

# Make obs names unique
x.obs_names_make_unique()

# Compute top 2000 HVGs (data is already log-transformed)
sc.pp.highly_variable_genes(x, n_top_genes=2000)

# Set X_hvg obsm key
hvg_data = x.X[:, x.var['highly_variable']]
if sparse.issparse(hvg_data):
    hvg_data = hvg_data.toarray()
x.obsm['X_hvg'] = hvg_data

# Write output
x.write('processed_data/replogle.h5ad')