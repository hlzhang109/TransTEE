# TransTEE

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is an official PyTorch implementation of the paper [Can Transformers be Strong Treatment Effect Estimators?](https://arxiv.org/abs/2202.01336)



## Abstract 
In this paper, we develop a general framework based on the Transformer architecture to address a variety of challenging treatment effect estimation (TEE) problems. Our methods are applicable both when covariates are tabular and when they consist of sequences (e.g., in text), and can handle discrete, continuous, structured, or dosage-associated treatments. While Transformers have already emerged as dominant methods for diverse domains, including natural language and computer vision, our experiments with **Trans**formers as **T**reatment **E**ffect **E**stimators (TransTEE) demonstrate that these inductive biases are also effective on the sorts of estimation problems and datasets that arise in research aimed at estimating causal effects. Moreover, we propose a propensity score network that is trained with TransTEE in an adversarial manner to promote independence between covariates and treatments to further address selection bias. Through extensive experiments, we show that TransTEE significantly outperforms competitive baselines with greater parameter efficiency over a wide range of benchmarks and settings.
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
The repository is under construction. If you find this repo useful, please cite: 
```
@misc{zhang2022transformers,
      title={Can Transformers be Strong Treatment Effect Estimators?}, 
      author={Yi-Fan Zhang and Hanlin Zhang and Zachary C. Lipton and Li Erran Li and Eric P. Xing},
      year={2022},
      eprint={2202.01336},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```