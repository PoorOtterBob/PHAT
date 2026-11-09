<div align="center">

# PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting

[![Venue ICLR%202026](https://img.shields.io/badge/Venue-ICLR%202026-4b44ce)](https://iclr.cc/virtual/2026/poster/10007638)
[![arXiv](https://img.shields.io/badge/arXiv-2602.00654-b31b1b.svg)](https://arxiv.org/abs/2602.00654)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-ee4c2c.svg)](https://pytorch.org/)

<p align="center">
  <a href="https://arxiv.org/abs/2602.00654">Paper</a> |
  <a href="#11-coding-framework">Code framework</a> |
  <a href="#-citation">Citation</a> |
  <a href="#-license">License</a>
</p>

</div>

This is the repository of our ICLR 2026 paper. While existing multivariate time series forecasting models have advanced significantly in modeling periodicity, they largely neglect the periodic heterogeneity common in real-world data, where variables exhibit distinct and dynamically changing periods. To effectively capture this periodic heterogeneity, we propose **PHAT** (**P**eriod **H**eterogeneity-**A**ware **T**ransformer). Specifically, PHAT arranges multivariate inputs into a three-dimensional "**periodic bucket**" tensor, where the dimensions correspond to variable group characteristics with similar periodicity, time steps aligned by phase, and offsets within the period. By restricting interactions within buckets and masking cross-bucket connections, PHAT effectively avoids interference from inconsistent periods. We also propose a positive-negative attention mechanism, which captures periodic dependencies from two perspectives: periodic alignment and periodic deviation. Additionally, the periodic alignment attention scores are decomposed into positive and negative components, with a modulation term encoding periodic priors. This modulation constrains the attention mechanism to more faithfully reflect the underlying periodic trends. A mathematical explanation is provided to support this property. We evaluate PHAT comprehensively on 14 real-world datasets against 18 baselines, and the results show that it significantly outperforms existing methods, achieving highly competitive forecasting performance.

<img src='PHAT.png' alt='Motivation of PHAT'>

## 1. Introduction about the code
### 1.1 Coding Framework
All of our experiments are running on the [benchmark](https://github.com/decisionintelligence/TFB) framework. To run PHAT, you need to configure your environment and datasets according to their requirements. Then put this repository in the `ts_benchmark/baselines/`.

<br>

## 2. Environmental Requirments
The experiment requires the same environment as [benchmark](https://github.com/decisionintelligence/TFB).

## 📄 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{ma2026phat,
  title     = {{PHAT}: Modeling Period Heterogeneity for Multivariate Time Series Forecasting},
  author    = {Ma, Jiaming and Huang, Qihe and Ma, Haofeng and Wang, Guanjun and Huang, Sheng and
               Zhou, Zhengyang and Wang, Pengkun and Wang, Binwu and Wang, Yang},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.00654}
}
```

## 🙏 Acknowledgements

Our experiments are built on the [TFB](https://github.com/decisionintelligence/TFB) benchmark framework.

## 📜 License

The code in this repository is released under the [MIT License](LICENSE).
