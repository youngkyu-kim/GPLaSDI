# multi-stage LaSDI

multi-stage LaSDI provides tools for building reduced-order models from full order simulations using latent-space dynamics identification. This repo can be used to reproduce the examples from "mLaSDI: Multi-stage latent space dynamics identification" [[Preprint]](https://arxiv.org/abs/2506.09207)



## Getting Started

After installing GPLaSDI, all mLaSDI experiments in this directory should be able to run. A user only needs to run

```bash
python train_model.py
```

in the directory for the appropriate problem. This will train first and second stage models for mLaSDI, along with producing figures in the paper. Code for the Vlasov models still needs to be included, but is similar to the provided examples