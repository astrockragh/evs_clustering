# `evs_clustering`
A python module for calculating Extreme Value Statistics (EVS) of galaxy stellar mass distributions in **high-density environments**.

This repository contains code and notebooks replicating the work by Jespersen, Carnall, and Lovell 2025 (see [the ApJL](https://doi.org/10.3847/2041-8213/adeb7c) or [the arXiV paper](https://doi.org/10.3847/2041-8213/adeb7c)), with a focus on mdoelling the extreme masses of ultra-massive quiescent galaxies seen in the EXCELS high-redshift survey. The goal is to understand how their environmental overdensity (clusters) affects the statistics the most massive galaxies by coupling stellar mass functions (SMFs) to density percentiles through clustering statistics and cosmic variance.

---

## 🚀 Overview

We model the maximum stellar mass expected in a given survey by:

1. **Constructing overdensity-dependent SMFs**  
   Based on a baryon conversion efficiency and halo mass function (HMF), with variance from cosmic variance modeled via a gamma distribution.

2. **Estimating density percentiles using EVS**  
   Using EVS on uniformly distributed percentile variables to estimate the expected maximum density percentile in a survey.

3. **Convolving EVS and SMFs**  
   Predicting the maximum stellar mass given the expected most extreme environment sampled in a survey.

A visual demo of how clustering impacts the SMF shape is included via an animation in the notebooks.

There are two demonstration notebooks included, a minimal working example in `evs_clustering_demo_minimal.ipynb` for those who just want the code to work, and `evs_clustering_demo_full.ipynb` for those who really want to understand how and why the code works. `evs_clustering_demo_full.ipynb` also contains some fun animations. Have fun using the code and let me know if anything doesn't work!

---

## 📂 Repository Structure

```bash
evs_clustering/
├── data/                       # Contains input HDF5 files (excluded from git)
├── graphics/                   # Output plots, figures, and demo visuals
├── dfs/                        # DataFrames used or produced by analysis
├── evs_clustering_demo_full.ipynb   # Full notebook with visual explanations
├── evs_clustering_demo_minimal.ipynb # Minimal demonstration
├── README.md
└── .gitignore

For completeness, [Lovell et al. 2023](https://academic.oup.com/mnras/article/518/2/2511/6823705) provides a simple introduction to EVS in unclustered environments.
