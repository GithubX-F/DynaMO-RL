<div align="center">

# DynaMO: Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization

<p align="center">
  <strong>DynaMO</strong> · Official Implementation
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2602.19208">
    <img src="https://img.shields.io/badge/Paper-arXiv-red.svg" alt="Paper" />
  </a>
  <a href="https://github.com/verl-project/verl">
    <img src="https://img.shields.io/badge/Framework-verl-blueviolet.svg" alt="Framework: verl" />
  </a>
  <img src="https://img.shields.io/badge/Domain-Dynamic%20Rollout%20%7C%20LLM%20Reasoning-orange.svg" alt="Domain: Dynamic Rollout | LLM Reasoning" />
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="PRs Welcome" />
</p>

</div>

## 📖 Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for Large Language Model (LLM) reasoning, yet current methods face key challenges in resource allocation and policy optimization dynamics: (i) uniform rollout allocation ignores gradient variance heterogeneity across problems, and (ii) the softmax policy structure causes gradient attenuation for high-confidence correct actions, while excessive gradient updates may destabilize training. Therefore, we propose **DynaMO**, a theoretically-grounded dual-pronged optimization framework. *At the sequence level*, we prove that uniform allocation is suboptimal and derive variance-minimizing allocation from the first principle, establishing Bernoulli variance as a computable proxy for gradient informativeness. *At the token level*, we develop gradient-aware advantage modulation grounded in theoretical analysis of gradient magnitude bounds. Our framework compensates for gradient attenuation of high-confidence correct actions while utilizing entropy changes as computable indicators to stabilize excessive update magnitudes. Extensive experiments conducted on a diverse range of mathematical reasoning benchmarks demonstrate consistent improvements over strong RLVR baselines, validating the effectiveness of DynaMO across various LLM scales and problem difficulties.

## 🎯 Key Contributions

- **Variance-Minimizing Rollout Allocation**: We prove that uniform allocation is suboptimal and derive a dynamic rollout allocation strategy that explicitly balances the informativeness-noise trade-off by minimizing gradient variance, using Bernoulli variance as a lightweight proxy.

- **Gradient-Aware Advantage Modulation**: We establish the gradient-entropy relationship through theoretical analysis, enabling a token-level mechanism that compensates for gradient attenuation in high-confidence correct actions and stabilizes excessive update magnitudes using entropy changes as an indicator.

- **Superior Performance**: Extensive experiments across six benchmarks (AIME24, AIME25, AMC23, MATH500, Minerva, Olympiad) and three LLM scales (1.5B, 7B, 14B) demonstrate consistent improvements, with comprehensive ablations validating each component.

## 🎨 Overview

<div align="center">
  <img src="docs/images/overview.png" alt="DynaMO Overview" width="100%">
  <p><em>Figure 1: Overview of DynaMO framework operating at both sequence and token levels.</em></p>
</div>

## 🚀 Quick Start

### Installation

This implementation is based on [verl](https://github.com/volcengine/verl), a flexible and efficient RLHF framework. Please follow the verl installation guide first.

```bash
# Clone the repository
git clone https://github.com/your-repo/DynaMO-RL.git
cd DynaMO-RL

# Install verl dependencies (see verl documentation for details)
pip install -r requirements.txt
```

### Running DynaMO

We provide example scripts for training with DynaMO. For example, to train a 7B model:

```bash
bash examples/dynamo_trainer/run_qwen2.5-math-7b.sh
```

## 📊 Experimental Results

### Main Results

We evaluate DynaMO on multiple mathematical reasoning benchmarks including AIME24, AIME25, AMC23, MATH500, Minerva, and OlympiadBench. The following table shows the comprehensive comparison with competitive baselines:

#### Comparison of benchmark results across Qwen2.5-Math-1.5B and Qwen2.5-Math-7B

| Method | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | **Avg.** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | P@1 / P@32 | P@1 / P@32 | P@1 / P@32 | P@1 / P@32 | P@1 / P@32 | P@1 / P@32 | P@1 / P@32 |
| **Qwen2.5-Math-1.5B** | | | | | | | |
| GRPO | 13.2 / 32.3 | 7.6 / 31.5 | 56.0 / 90.0 | 54.4 / 79.2 | 17.2 / 42.8 | 25.6 / 47.0 | 29.0 / 53.8 |
| Clip-Higher | 12.4 / 34.7 | 6.4 / 30.6 | 50.6 / 89.9 | 56.8 / 80.2 | 16.8 / 41.3 | 26.4 / 46.8 | 28.2 / 53.9 |
| Entropy Loss | 12.6 / 33.7 | 5.8 / 28.4 | 55.6 / 86.9 | 56.3 / 78.5 | 17.6 / 43.6 | 25.4 / 46.4 | 28.9 / 52.9 |
| Fork Tokens | 9.4 / 32.0 | 5.9 / 31.4 | 52.5 / 85.6 | 54.3 / 74.2 | 16.6 / 36.8 | 25.5 / 45.2 | 27.4 / 50.9 |
| Entropy Advantages | 15.7 / 35.8 | 8.9 / 33.4 | 62.0 / 86.4 | **59.7** / 76.2 | 18.2 / 43.0 | 25.9 / 44.9 | 31.7 / 53.3 |
| Clip-COV | 13.5 / 36.4 | 6.6 / **34.4** | 59.5 / 89.7 | 57.6 / 75.6 | 15.8 / **44.3** | 25.8 / 47.6 | 29.8 / 54.7 |
| KL-COV | 12.6 / 33.9 | 9.0 / 33.4 | 55.8 / 91.3 | 54.2 / 78.1 | 14.8 / 40.3 | 25.4 / **48.1** | 28.6 / 54.2 |
| W-REINFORCE | 15.3 / 35.3 | 8.5 / 31.7 | 63.0 / 85.7 | 56.7 / 77.7 | 18.2 / 40.3 | 24.4 / 46.2 | 31.0 / 52.8 |
| **DynaMO (Ours)** | **17.2** / **37.2** | **9.8** / 32.5 | **63.6** / **91.9** | 58.8 / **81.0** | **19.4** / 44.0 | **27.2** / 47.1 | **32.7** / **55.6** |
| **Qwen2.5-Math-7B** | | | | | | | |
| GRPO | 28.8 / 52.5 | 11.7 / 34.8 | 68.3 / 90.8 | 63.3 / 75.0 | 22.6 / 45.4 | 28.6 / 44.7 | 37.2 / 57.2 |
| Clip-Higher | 27.0 / 51.9 | 12.1 / 39.5 | 67.8 / 89.9 | 64.2 / 83.6 | 24.0 / 46.1 | 28.1 / 46.3 | 37.2 / 59.6 |
| Entropy Loss | 30.6 / 54.6 | 13.2 / 40.6 | 66.0 / 87.0 | 60.6 / 79.6 | 23.3 / 45.9 | 30.2 / 41.1 | 37.3 / 58.1 |
| Fork Tokens | 27.1 / 52.5 | 13.4 / 43.5 | 71.0 / 87.3 | 65.8 / 79.3 | 26.1 / 42.4 | 30.9 / 47.3 | 39.1 / 58.7 |
| Entropy Advantages | 27.5 / 49.7 | 9.4 / 39.2 | 67.9 / 85.2 | 65.3 / 83.3 | 23.7 / 43.7 | 30.4 / 47.3 | 37.4 / 58.1 |
| Clip-COV | 32.2 / 52.7 | 13.2 / 40.4 | 72.7 / 89.3 | 64.3 / 76.8 | 25.4 / 45.9 | 29.5 / 44.6 | 39.5 / 58.3 |
| KL-COV | 32.8 / 53.3 | 11.7 / 36.1 | 70.6 / 88.5 | 64.6 / 75.3 | 24.5 / 39.9 | 30.2 / 44.2 | 39.1 / 56.2 |
| W-REINFORCE | 31.8 / 55.4 | 14.3 / 41.0 | 72.5 / 89.8 | 64.9 / **84.0** | 26.4 / **49.5** | 30.9 / 46.7 | 40.1 / 61.1 |
| **DynaMO (Ours)** | **34.4** / **59.0** | **15.4** / **46.8** | **74.4** / **92.9** | **66.4** / **84.0** | **27.3** / 47.2 | **31.6** / **50.1** | **41.6** / **63.3** |

**Key Findings:**
- **1.5B Model**: DynaMO outperforms GRPO and other baselines significantly in both P@1 and P@32.
- **7B Model**: DynaMO achieves the best performance across almost all benchmarks, demonstrating superior scalability.
- **Bold** indicates best performance.

## 🔬 Algorithm Overview

### 1. Dynamic Rollout Allocation (Sequence Level)

We derive the optimal rollout allocation by minimizing the total gradient estimation variance. The optimal number of rollouts $n_i^*$ for prompt $q_i$ is proportional to its gradient standard deviation $\sigma_i$:

$$
n_i^* = B \cdot \frac{\sigma_i}{\sum_{k=1}^N \sigma_k}
$$

We use the Bernoulli variance $P_i$ as a practical proxy for $\sigma_i$, which can be estimated from historical success statistics:

$$
P_i = \frac{k_i(G_i - k_i)}{G_i(G_i - 1)}
$$

where $k_i$ is the number of correct responses and $G_i$ is the total rollouts generated so far.

### 2. Gradient-Aware Advantage Modulation (Token Level)

We introduce a modulation factor to the advantage function:

$$
A_{i,t}^{\text{final}} = A_{i,t} \cdot \beta_{i,t}^{\text{comp}} \cdot \beta_{i,t}^{\text{stab}}
$$

#### Gradient Compensation ($\beta^{\text{comp}}$)
To mitigate gradient attenuation for high-confidence correct actions, we use an entropy-aware compensation factor:

$$
\beta_{i,t}^{\text{comp}} = \mathbb{I}[A_{i,t} > 0] \cdot \left( 1 + \alpha \cdot \frac{\mathcal{H}_{\max} - \mathcal{H}_{i,t}}{\mathcal{H}_{\max} - \mathcal{H}_{\min}} \right) + \mathbb{I}[A_{i,t} \leq 0]
$$

#### Update Magnitude Stabilization ($\beta^{\text{stab}}$)
To prevent excessive updates, we use entropy change $\Xi_{i,t} = |\Delta \mathcal{H}_{i,t}|$ as an instability indicator:

$$
\beta_{i,t}^{\text{stab}} = f\left( \frac{\Xi_{i,t}}{\max_j \Xi_{j,t}} \right)
$$

where $f(\cdot)$ is a sigmoid-based decay function.


## 🔧 Implementation Details

- **Dynamic Rollout Allocation**: Implemented in `recipe/dynamo/dynamo_ray_trainer.py` via `get_rollout_n_per_prompt` function.
- **Gradient-Aware Advantage Modulation**: Implemented in `verl/workers/actor/dp_actor.py` inside `update_policy` and `_compute_entropy_estimation`.

## 📚 Citation

If you find DynaMO helpful in your research or applications, please consider citing our paper:

```bibtex
@article{dynamo,
  title={How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization},
  author={Fang, Yangyi and Lin, Jiaye and Fu, Xiaoliang and Qin, Cong and Shi, Haolin and Hu, Chaowen and Pan, Lu and Zeng, Ke and Cai, Xunliang},
  journal={arXiv preprint arXiv:2602.19208},
  year={2026}
}
```

## 🙏 Acknowledgments

This implementation is built on top of [verl](https://github.com/volcengine/verl), a flexible and efficient RLHF framework. We thank the verl community for their excellent infrastructure.

## 📝 License

This project follows the same license as verl. Please refer to the verl repository for license details.

---

<div align="center">

**⭐ If DynaMO helps your research or applications, please give us a star! ⭐**

**Note**: This is the official implementation of DynaMO. For more details, please refer to our paper ❤️.

</div>