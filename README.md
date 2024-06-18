
<p align="center">
    <br>
    <img src="https://github.com/yeshenpy/VEB-RL/assets/43668853/7e66cbfb-6159-4acf-ba5c-d09418124ed2" width="300"/>
    <br>
<p>

  
# :beginner: Value-Evolutionary-Based Reinforcement Learning (VEB-RL)
(ICML 2024) The official code for VEB-RL from [Value-Evolutionary-Based Reinforcement Learning](https://openreview.net/forum?id=h9LcbJ3l9r) by [Pengyi Li](https://yeshenpy.github.io/).


## :triangular_flag_on_post: Method

**VEB-RL** is a hybrid framework specifically designed for value-based reinforcement learning methods. VEB-RL integrates genetic algorithms (GA) and cross-entropy method (CEM), using TD error as fitness for more accurate value function approximation. We also propose the Elite Interaction Mechanism to improve sample quality. VEB-RL significantly enhances value-based RL across various tasks.

![image](https://github.com/yeshenpy/VEB-RL/assets/43668853/ab222e78-9ef1-49b8-9df3-9b104b366712)

<span style="color: red;"> </span>

> [!TIP]
> 🔥 🔥 🔥 If you are interested in ERL for policy search or other hybrid algorithms combining EA and RL, we **strongly recommend** reading our survey paper: **[Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey on Hybrid Algorithms](https://arxiv.org/abs/2401.11963)**. It provides a comprehensive and accessible overview of research directions and classifications suitable for researchers with various backgrounds.


## 🙏 Citation

If you do find our paper or the repository helpful (or if you would be so kind as to offer us some encouragement), please consider kindly giving a star, and citing our paper.
```

@inproceedings{li2023value,
  title={Value-Evolutionary-Based Reinforcement Learning},
  author={Li, Pengyi and Jianye, HAO and Tang, Hongyao and Zheng, Yan and Barez, Fazl},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2023}
}


```


## 🛠️ Instructions


You need to create a [Weights & Biases](https://wandb.ai) account for visualizing results, and you should already have conda installed.

First, we create an environment based on the provided requirements.txt:

```
conda create --name VEBRL --file requirements.txt
```

Activate the environment:

```
conda activate VEBRL
```

Then enter either the GA_VEB folder or the CEM_VEB folder and directly run run.sh

```
cd ./GA_VEB or cd ./CEM_VEB
chmod 777 ./run.sh
./run.sh
```

The specific hyperparameter settings need to be adjusted according to the original paper.


## :beginner: License & Acknowledgements

VEB-RL is licensed under the MIT license.

## ✉ Contact

For any questions, please feel free to email `lipengyi@tju.edu.cn`.

