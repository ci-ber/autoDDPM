# AutoDDPM
## Mask, Stitch, and Re-Sample: Enhancing Robustness and Generalizability in Anomaly Detection through Automatic Diffusion Models
</h1>
  <p align="center">
    <a href="https://ci.bercea.net">Cosmin Bercea</a> •
    Michael Neumayr •
    <a href="https://aim-lab.io/author/daniel-ruckert/">Daniel Rueckert </a> •
    <a href="https://compai-lab.github.io/author/julia-a.-schnabel/">Julia A. Schnabel </a>
  </p>
  
<h4 align="center"><a href="https://ci.bercea.net/project/autoddpm/">Project Website</a>

Published at ICML IMLH 2023: 
- https://openreview.net/pdf?id=kTpafpXrqa
- https://arxiv.org/abs/2305.19643

![method](https://github.com/ci-ber/autoDDPM/assets/106509806/54bebddf-d074-4eb9-82f3-3115f8625fc7)

## Citation

If you find our work useful, please cite our paper:
```
@article{bercea2023mask,
title={Mask, Stitch, and Re-Sample: Enhancing Robustness and Generalizability in Anomaly Detection through Automatic Diffusion Models},
author={Bercea, Cosmin I and Neumayr, Michael and Rueckert, Daniel and Schnabel, Julia A},
journal={arXiv preprint arXiv:2305.19643},
year={2023}
}
```

## Abstract

The introduction of diffusion models in anomaly detection has paved the way for more effective and accurate image reconstruction in pathologies. However, the current limitations in controlling noise granularity hinder diffusion models' ability to generalize across diverse anomaly types and compromise the restoration of healthy tissues. To overcome these challenges, we propose AutoDDPM, a novel approach that enhances the robustness of diffusion models. 

AutoDDPM utilizes diffusion models to generate initial likelihood maps of potential anomalies and seamlessly integrates them with the original image. Through joint noised distribution re-sampling, AutoDDPM achieves harmonization and in-painting effects. Our study demonstrates the efficacy of AutoDDPM in replacing anomalous regions while preserving healthy tissues, considerably surpassing diffusion models' limitations. 

It also contributes valuable insights and analysis on the limitations of current diffusion models, promoting robust and interpretable anomaly detection in medical imaging - an essential aspect of building autonomous clinical decision systems with higher interpretability.


## Setup and Run

The code is based on the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: https://github.com/compai-lab/iml-dl

#### Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### Clone repository

```bash
git clone https://github.com/ci-ber/autoDDPM.git
cd autoDDPM
```

#### Install requirements

```bash
pip install -r requirements.txt
```

#### Run the pipeline

Run the main script with the corresponding config like this:

```bash
python core/Main.py --config_path ./projects/autoddpm/autoddpm.yaml
```

Refer to the autoddpm.yaml for the default configuration. Store the pretrained model from LINK into the specified directory to skip the training part.

By default, reconstructed images (from the first masking part of the pipeline) and inpainted images (after the second stitching and resampling part of the pipeline) are stored so that one can work on the parts of the pipeline in a modular way.





