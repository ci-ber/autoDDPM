# AutoDDPM
## Mask, Stitch, and Re-Sample: Enhancing Robustness and Generalizability in Anomaly Detection through Automatic Diffusion Models

Code for https://arxiv.org/abs/2305.19643

## Abstract

The introduction of diffusion models in anomaly detection has paved the way for more effective and accurate image reconstruction in pathologies. However, the current limitations in controlling noise granularity hinder diffusion models' ability to generalize across diverse anomaly types and compromise the restoration of healthy tissues. To overcome these challenges, we propose AutoDDPM, a novel approach that enhances the robustness of diffusion models. 

AutoDDPM utilizes diffusion models to generate initial likelihood maps of potential anomalies and seamlessly integrates them with the original image. Through joint noised distribution re-sampling, AutoDDPM achieves harmonization and in-painting effects. Our study demonstrates the efficacy of AutoDDPM in replacing anomalous regions while preserving healthy tissues, considerably surpassing diffusion models' limitations. 

It also contributes valuable insights and analysis on the limitations of current diffusion models, promoting robust and interpretable anomaly detection in medical imaging - an essential aspect of building autonomous clinical decision systems with higher interpretability.


## Setup and Run

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