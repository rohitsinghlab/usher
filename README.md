# USHER: A Unified Strategy for Shifting Towards Established Representations in Foundation Models

A comprehensive toolkit for adapting out-of-distribution data to foundation models through linear optimal transport.

## Overview

USHER enables foundation models to process diverse, out-of-distribution data sources without retraining or fine-tuning. Instead of modifying the model, USHER learns interpretable linear transformations that shift new data distributions towards the established representations that foundation models expect. This approach preserves model integrity while expanding their operational scope.
![USHER Overview](figs/USHER_overview.jpg)

## Key Features

- **Multiple alignment methods**: Optimal Transport (OT), Gromov-Wasserstein (GW), Fused Gromov-Wasserstein (FGW)
- **Spatial windowing**: Memory-efficient processing of large spatial datasets
- **Distribution matching**: Sliced Wasserstein Distance (SWD) and Maximum Mean Discrepancy (MMD) for improved alignment
- **GPU acceleration**: Full PyTorch implementation with CUDA support
- **Flexible metrics**: Support for both cosine and Euclidean distance metrics

## Installation

```bash
# Required packages
pip install torch torchvision
pip install scanpy anndata
pip install POT  # Python Optimal Transport
pip install numpy pandas scipy
pip install matplotlib seaborn
pip install tqdm
```

## Quick Start

```python
import anndata as ad
from src.run_usher import align_features_fgw

# Load your data
adata_source = ad.read_h5ad('source.h5ad')
adata_target = ad.read_h5ad('target.h5ad')

# Run alignment
model, mapping, concat_adata, T_full, feature_mean, feature_std = align_features_fgw(
    adata_source,
    adata_target,
    sampling_strategy='spatial',  # or 'celltype'
    e_step_method='fgw',  # 'ot', 'gw', or 'fgw'
    metric='cosine',  # or 'euclidean'
    n_iters=10,
    device='cuda'
)

# Save the model for later use
from src.model_utils import save_alignment_model
save_alignment_model(model, 'alignment_model.pt', feature_mean, feature_std)
```

## Example Notebook

For a complete walkthrough, see the [scGPT example notebook](example_notebooks/scGPT_example.ipynb) which demonstrates aligning Xenium spatial transcriptomics data to scRNA-seq using scGPT embeddings.

### Dataset Download

The example datasets can be downloaded from Google Drive:
- **[usher_datasets](https://drive.google.com/drive/folders/1U_VW--4BaTaxVQdYLuU1wOwg7aUQmkRU)** - Contains reference scRNA-seq dataset `scRNAseq_scGPT.h5ad` (6.65 GB) and target Xenium spatial transcriptomics dataset `xenium_scGPT.h5ad` (465.5 MB) used in manuscript. The USHER model is under the `scGPT_example/alignment_model.pt`.

Place the downloaded files in the `datasets/` directory.

## Main Components

### Core Alignment Module
- `run_usher.py`: Main alignment function with E-M algorithm

### Utility Modules
- `e_step_utils.py`: Transport computation and optimal transport utilities
- `m_step_utils.py`: Model training and distribution matching losses (including SWD and MMD)
- `model_utils.py`: Model saving, loading, and application to new data
- `spatial_utils.py`: Spatial windowing and k-NN utilities
- `celltype_utils.py`: Cell type-based sampling strategies
- `plot_utils.py`: Visualization utilities



## Algorithm Overview

The alignment process follows an Expectation-Maximization approach:

1. **E-step**: Compute optimal transport between source and target cells
   - Uses Sinkhorn algorithm for efficient computation
   - Supports OT, GW, or FGW formulations
   - Incorporates spatial constraints for windowed processing

2. **M-step**: Learn feature transformation
   - Neural network learns to map source features to target space
   - Multiple loss components:
     - Cross-domain alignment loss
     - Structure preservation loss
     - Distribution matching (variance loss/preservation)



## Data Format

Input data should be AnnData objects with:
- Expression matrix in `.X`
- Spatial coordinates in `.obsm['X_spatial']` (for spatial data)
- Cell type probabilities in `.obsm['X_celltype_prob']` (optional)
- Cell metadata in `.obs`
- Gene metadata in `.var`

## Citation

If you use this code, please cite:
@article{pratapa-2025, 
	author = {Pratapa, Aditya and Tata, Purushothama Rao and Singh, Rohit},
	journal = {bioRxiv (Cold Spring Harbor Laboratory)},
	month = {11},
	title = {{USHER: Guiding Foundation Model Representations through Distribution Shifts}},
	year = {2025},
	doi = {10.1101/2025.11.20.689462},
	url = {[https://doi.org/10.1101/2025.11.20.689462](https://doi.org/10.1101/2025.11.20.689462)},
}
## License

CC BY NC SA 4.0

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


## Contact

For questions and issues, please open an issue on GitHub or contact the maintainers.
