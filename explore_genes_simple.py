"""
Simple script to explore gene lists in raw and processed data
Avoids the filtered files that have compatibility issues
"""
import anndata as ad
import pandas as pd

print(f"anndata version: {ad.__version__}\n")

# Load raw data
print("="*70)
print("LOADING RAW DATASETS")
print("="*70)

raw_files = {
    'hepg2': 'data/GSE264667_hepg2_raw_singlecell_01.h5ad',
    'jurkat': 'data/GSE264667_jurkat_raw_singlecell_01.h5ad',
    'k562': 'data/K562_essential_normalized_singlecell_01.h5ad',
    'rpe1': 'data/rpe1_normalized_singlecell_01.h5ad'
}

raw_data = {}
for name, path in raw_files.items():
    print(f"\nLoading {name}...")
    raw_data[name] = ad.read_h5ad(path)
    print(f"  Shape: {raw_data[name].shape}")

# Gene overlap analysis
print("\n" + "="*70)
print("GENE OVERLAP ANALYSIS")
print("="*70)

gene_sets = {name: set(adata.var_names) for name, adata in raw_data.items()}
common_genes = set.intersection(*gene_sets.values())

print(f"\nGenes per dataset:")
for name, genes in gene_sets.items():
    print(f"  {name:10s}: {len(genes):,} genes")

print(f"\nCommon genes across ALL datasets: {len(common_genes):,}")
print(f"\nFirst 20 common genes:")
print(sorted(list(common_genes))[:20])

# Load processed data
print("\n" + "="*70)
print("LOADING PROCESSED DATA")
print("="*70)

processed = ad.read_h5ad('processed_data/replogle.h5ad')
print(f"\nProcessed data: {processed.shape}")
print(f"  Cells: {processed.n_obs:,}")
print(f"  Genes: {processed.n_vars:,}")

# Compare
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

processed_genes = set(processed.var_names)
print(f"\nTotal raw cells: {sum(a.n_obs for a in raw_data.values()):,}")
print(f"Processed cells: {processed.n_obs:,}")
print(f"\nCommon genes (raw): {len(common_genes):,}")
print(f"Processed genes: {len(processed_genes):,}")
print(f"Are they the same? {processed_genes == common_genes}")

# HVG analysis
if 'highly_variable' in processed.var.columns:
    n_hvg = processed.var['highly_variable'].sum()
    print(f"\nHighly Variable Genes: {n_hvg:,} ({100*n_hvg/len(processed.var):.1f}%)")
    
if 'X_hvg' in processed.obsm.keys():
    print(f"X_hvg shape: {processed.obsm['X_hvg'].shape}")

print("\n" + "="*70)
print("DONE")
print("="*70)
