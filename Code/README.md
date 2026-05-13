# title:  Relative Wasserstein Spatial Depth for Cluster Number Selection in Distributional Data

This repository contains the workflow of implementing Relative Wasserstein Spatial Depth (RWSD) for selecting the number of clusters K in Wasserstein K-center clustering.

> **Anonymous submission.** All author and institution information has been removed in compliance with the double-blind review policy.

## Overview

The repository provides a complete algorithmic toolkit for clustering probability distributions in the 2-Wasserstein space. The framework consists of:

1. **Wasserstein K-means / K-median clustering** with PAM-based warm starts.
2. **Two cluster-center solvers**, exposed in two parallel implementations:
   - *Fixed-support* barycenters and Fréchet medians, computed via Sinkhorn / linear programming on a shared discrete support.
   - *Semi-free-support* barycenters and Fréchet medians, with a free atomic support that adapts to the data.
3. **Wasserstein Fréchet median via IRLS** (iteratively reweighted least squares), used to update the Wasserstein median of each cluster of distributions.
4. **RWSD-based cluster number selection**, and comparison to silhouette score and Davies–Bouldin Index.
5. **Theoretical guarantees**: a Pollard-type strong consistency result for Wasserstein K-means under compact-support assumptions.


## Repository Layout

```
.
├── README.md
├── .gitignore
│
├── Fixed-support version/         # Fixed-support K-means / K-median implementation
├── Semi-Free Support Version/     # Semi-free-support K-means / K-median implementation
│
├── rwsd_core.py                   # Simulation: main algorithms (clustering + selection criteria)
├── rwsd_experiments.py            # Simulation: data generating mechanism and experimental pipeline
├── run_multi_seed.py              # Simulation: multi-seed driver
│
├── mnist_core.py                  # MNIST experiment: all the analytic steps
│
├── cytof/                         # Flow Cytometry dataset experiment: all the analytic steps
└── P1_empirical_1000_norm.csv     # Preprocessed real dataset of Flow Cytometry
```

The `Fixed-support version/` and `Semi-Free Support Version/` folders contain different sub-types of Wasserstein K-median clustering. Each of them accommodates Wasserstein K-means and K-medians.

## Components

### Clustering implementations

| Folder | Support | K-means | K-median | Notes |
| --- | --- | --- | --- | --- |
| `Fixed-support version/` | Fixed (shared discrete grid) | ✓ | ✓ | Sinkhorn or LP backends |
| `Semi-Free Support Version/` | Free atomic support | ✓ | ✓ | Free-support barycenter / median |

### Simulation experiments

The three Python files below perform the synthetic-data simulations:

- `rwsd_core.py` — clustering algorithms, RWSD computation, Optimal K selection.
- `rwsd_experiments.py` — experimental procedure.
- `run_multi_seed.py` — top-level multi-seed runner that aggregates results across independent replications.


### MNIST experiment (real data)

`mnist_core.py` contains the pipeline to cluster MNIST images. 
The MNIST dataset to be downloaded by torchvision into `./data/` before running the above script; see the [MNIST database](http://yann.lecun.com/exdb/mnist/) for details.

### CyTOF experiment (real data)

The `cytof` file contains the experimental pipeline to analyze the AML Flow Cytometry data. See the within-file instructions for usage.

## Reproducing the Experiments

Each component is self-contained:

- **Synthetic simulations.** Run `python run_multi_seed.py` from the repository root. The runner reads experiment definitions from `rwsd_experiments.py` and aggregates results over the requested seed range.
- **MNIST.** Use the sample plans in `mnist_experiments.EXPERIMENTS` together with `mnist_core.validate_over_Ks_fixed(...)`. 
- **CyTOF.** See the code in the `cytof` file.
- **Fixed / semi-free clustering kernels.** The two files `Fixed-support version` and `Semi-Free Support Version` contain stand-alone scripts that can be invoked directly.

## Dependencies

```
python >= 3.9
numpy
scipy
pandas
matplotlib
joblib
scikit-learn
POT          # Python Optimal Transport
torchvision  # MNIST data loader
```

A standard scientific Python environment (e.g. `pip install numpy scipy pandas matplotlib joblib scikit-learn POT torchvision`) is sufficient.

## Notes on Anonymity

All scripts, comments, and data files have been scrubbed of author identifiers. Should any residual identifying information remain, it is unintentional; reviewers are kindly asked to disregard it.
