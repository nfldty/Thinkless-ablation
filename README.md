# Thinkless: LLM Learns When to Think

![intro](assets/intro.png)

> [**Thinkless: LLM Learns When to Think**](http://arxiv.org/abs/2505.13379)   
> *[Gongfan Fang](https://fangggf.github.io/), [Xinyin Ma](https://horseee.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*    
> *[xML Lab](https://sites.google.com/view/xml-nus), National University of Singapore*
  
<table>
<table>
  <thead>
  </thead>
  <tbody>
    <tr>
      <td>📄 <strong>Paper Link</strong></td>
      <td><a href="http://arxiv.org/abs/2505.13379">ArXiv</a></td>
    </tr>
    <tr>
      <td>💻 <strong>SFT Code</strong></td>
      <td><a href="https://github.com/VainF/Reasoning-SFT">VainF/Reasoning-SFT</a></td>
    </tr>
    <tr>
      <td>🤖 <strong>RL Model</strong></td>
      <td><a href="https://huggingface.co/Vinnnf/Thinkless-1.5B-RL-DeepScaleR">Thinkless-1.5B-RL-DeepScaleR</a></td>
    </tr>
    <tr>
      <td>🐣 <strong>Warmup Model</strong></td>
      <td><a href="https://huggingface.co/Vinnnf/Thinkless-1.5B-Warmup">Thinkless-1.5B-Warmup</a></td>
    </tr>
    <tr>
      <td>📊 <strong>Data for Warmup</strong></td>
      <td><a href="https://huggingface.co/datasets/Vinnnf/Hybrid-OpenThoughts2-1M-1.5B">Hybrid-OpenThoughts2-1M-1.5B</a></td>
    </tr>
    <tr>
      <td>📊 <strong>Data for RL</strong></td>
      <td><a href="https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset">agentica-org/DeepScaleR-Preview-Dataset</a></td>
    </tr>
  </tbody>
</table>

## Introduction

> ***Can LLMs learn when to think?***

We propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, \<short\> for concise responses and \<think\> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50\% - 90\%, significantly improving the computational efficiency of Reasoning Language Models.

## The Full Pipeline

![full_pipeline](https://github.com/user-attachments/assets/468c89da-80f4-4b95-9829-7b81f7f303e2)



## Installation

```bash
conda create -n thinkless python==3.10
conda activate thinkless

# For training
cd Thinkless
pip install torch==2.4.0 lm_eval==0.4.8 ray==2.45.0 # install lm_eval before verl to avoid conflict
pip install -e ./verl
pip install -e .
# https://github.com/vllm-project/vllm/issues/4392
pip install nvidia-cublas-cu12==12.4.5.8
```


## Reproducing reported results 

#### LM-Eval
This script will repeat the generation for 5 times using lm_eval. All results will be saved in `./eval_results`.
```bash
bash run_eval.sh
```

#### Extract answers for evaluation
We only use LM-Eval for generation but do not use the built-in answer extractor. Instead, we developed an [evaluation tool](scripts/eval) based on the prompts in [openai/simple-evals](https://github.com/openai/simple-evals). To obtain the final metrics, please run the following command:
```bash
bash scripts/eval/eval_all.sh YOUR_MODEL_PATH THE_EVAL_RESULTS_PATH
```
For example, to evaluate the results under *eval_results/Vinnnf__Thinkless-1.5B-RL-DeepScaleR*, run the following command:
```bash
bash scripts/eval/eval_all.sh Vinnnf/Thinkless-1.5B-RL-DeepScaleR eval_results/Vinnnf__Thinkless-1.5B-RL-DeepScaleR
```

**Reproduction results** (Pass@1 and average number of generated tokens):

| Benchmark   | Pass@1 | Avg #Tokens |
|:-----------|------:|------------:|
| AIME 2024  | 0.2606 | 7221 |
| Minerva Algebra | 0.9423 | 1124 |
| Math-500   | 0.8193 | 2531 |
| GSM8k      | 0.8375 |  627 |

The reproduced results closely match the reported numbers in the paper. Across all four benchmarks, the difference in Pass@1 is within approximately 0.01-1.3%, and average token counts are nearly identical. This confirms that the performance reported in the paper is reproducible using the released checkpoint and evaluation pipeline.

## RL Training

### 1. Prepare the DeepScaleR Dataset for RL
```bash
scripts/data/prepare_deepscaler_for_RL.py
```
```
AIME-24 val data size: 30
DeepScaler data size: 40315
```

### 2. Run the RL script
```bash
bash run_train_rl.sh
```
We can tune the following hyperparameters in [`scripts/rl/thinkless_1.5b_deepscaler.sh`](scripts/rl/thinkless_1.5b_deepscaler.sh) to obtain a good performance.
```bash
# Whether to enable std normalization in advantage computing (False for Dr. GRPO)
algorithm.std_normalizer=False \ 
# The weight of decoupled control token loss. A higher value will lead to rapid convergence of mode selection.
actor_rollout_ref.actor.thinkless_alpha=0.001 \ 
# Increase this if you want to encourage thinking mode
thinkless_rewards.correct_think_reward=0.5 \ 
```

#### Resume
You can resume training from a checkpoint by modifying the `run_train_rl.sh`:
```bash
export MODEL_PATH="PATH_TO_YOUR_MODEL"
./scripts/rl/thinkless_1.5b_deepscaler.sh --model $MODEL_PATH
```

It's also recommended to have a new exp name in `scripts/rl/thinkless_1.5b_deepscaler.sh`:
```
trainer.experiment_name='Thinkless-1.5b-DeepScaleR-Resume' \
```

### Ablation: `correct_think_reward=0.9`

We resumed training from a checkpoint and continued for 50 steps with **`correct_think_reward=0.9`** (increased from the default 0.5) to encourage more use of think mode. Results (Pass@1 / Avg #Tokens):

| Benchmark   | Pass@1 | Avg #Tokens |
|:-----------|------:|------------:|
| AIME 2024  | 0.2801 | 8587 |
| Minerva Algebra | 0.9533 | 1586 |
| Math-500   | 0.8329 | 3004 |
| GSM8k      | 0.8510 | 1251 |

Higher `correct_think_reward` improves accuracy on AIME, Minerva, and GSM8k at the cost of more tokens (model uses think mode more often).


## Acknowledgements

* The RL part is based on the [agentica-project/rllm](https://github.com/agentica-project/rllm) (Previously named DeepScaleR).
* The warmup training is powered by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). We will release the a llama-factory version in the future.
* The following datasets are used in our experiments:
  * [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset): For RL training.
  * [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M/viewer/default/train?views%5B%5D=train): For warmup training.

## Bibtex
If you find this repository helpful, please consider citing our work:
```bibtex
@article{fang2025thinkless,
  title={Thinkless: LLM Learns When to Think},
  author={Fang, Gongfan and Ma, Xinyin and Wang, Xinchao},
  journal={Advances in neural information processing systems},
  year={2025}
}
```
