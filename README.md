# TransTEE

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is an official PyTorch implementation of the paper [Exploring Transformer Backbones for Heterogeneous Treatment Effect Estimation](https://arxiv.org/abs/2202.01336)



## Abstract 
Neural networks (NNs) are often leveraged to represent structural similarities of potential outcomes (POs) of different treatment groups to obtain better finite-sample estimates of treatment effects. However, despite their wide use, existing works handcraft treatment-specific (sub)network architectures for representing various POs, which limit their applicability and generalizability. To remedy these issues, we develop a framework called Transformers as Treatment Effect Estimators (TransTEE) where attention layers govern interactions among treatments and covariates to exploit structural similarities of POs for confounding control. Using this framework, through extensive experiments, we show that TransTEE can: (1) serve as a general-purpose treatment effect estimator to significantly outperform competitive baselines on a variety of challenging TEE problems (e.g., discrete, continuous, structured, or dosage-associated treatments.) and is applicable both when covariates are tabular and when they consist of structural data (e.g., texts, graphs); (2) yield multiple advantages: compatibility with propensity score modeling, parameter efficiency, robustness to continuous treatment value distribution shifts, interpretability in covariate adjustment, and real-world utility in debugging pre-trained language models.
<div align=center>

![A motivating example. **Prev** denotes previous infection condition and **BP** denotes blood pressure. A corresponding causal graph where shady nodes denote observed variable and white node denotes hidden outcome. TransTEE adjusts proper covariate sets **Prev,BP** with attention which is visualized via a heatmap.](model.png)
</div>

### Dependencies
Dependencies of different settings are listed in each subfolder.

#### Continuous Treatments
./Continuous

#### Continuous Dosage
./Dosage

#### Structured Treatments
./Structured

#### Empirical Study on Pre-trained Language Models
./CausalM

### Citation 
```
@misc{zhang2022transformers,
      title={Exploring Transformer Backbones for Heterogeneous Treatment Effect Estimation}, 
      author={Yi-Fan Zhang and Hanlin Zhang and Zachary C. Lipton and Li Erran Li and Eric P. Xing},
      year={2022},
      eprint={2202.01336},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
