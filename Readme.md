This is an official PyTorch implementation of NeurIPS 2025 paper "A Color-Structure Prior Guided Visual Semantic Fusion Network for Low-Light Image Enhancement."

This is a raw version at the moment; a tweaked version will be released online after the paper is accepted!

## Specification of dependencies

### Dependencies and Installation

- Python 3.7.0
- Pytorch 1.13.1

- (1) Create Conda Environment

```bash
conda create --name  CSPVSF python=3.7.0
conda activate  CSPVSF
```

- (2) Install Dependencies

```bash
cd  CSPVSF
pip install -r requirements.txt
```

### Data Preparation (Both the datasets and pretrained weights are publicly available and do not contain any information that could reveal the authors' identities.)

LOLv1, LOLv2-Real, LOLv2-Synthetic, DICM, LIME, MEF, NPE, and VV are all publicly available datasets.
Then, put them in the following folder:

<details close> <summary>datasets (click to expand)</summary>

```
├── datasets
	├── DICM
	├── LIME
	├── LOLdataset
		├── our485
			├──low
			├──high
		├── eval15
			├──low
			├──high
	├── LOLv2
		├── Real_captured
			├── Train
				├── Low
				├── Normal
			├── Test
				├── Low
				├── Normal
		├── Synthetic
			├── Train
				├── Low
				├── Normal
			├── Test
				├── Low
				├── Normal
	├── MEF
	├── NPE
	├── VV
```
</details>


## Pretrained weights
Download the pretrained weights of DiNOv2-B to pretrain
```bash
mkdir pretrain && cd pretrain
## DiNOv2-B
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
```

# Quick Start

To train  CSPVSF, modify the script according to your requirements and run it:

```
python train.py
```


## Weights
After acceptance of our paper, we will open-source our model weights.


## Results
The metrics of SCP-VSF on paired datasets are shown in the following table:
| metrics | LOLv1  | LOLv2-Real  | LOLv2-Synthetic   | 
| ------- | ----- | ----- | ----- | 
| PSNR    | 25.84 | 24.94  | 26.09  |
| SSIM | 0.893 | 0.883 | 0.942 | 


Performance on five unpaired datasets are shown in the following table:
| metrics | DICM  | LIME  | MEF   | NPE   | VV    |
| ------- | ----- | ----- | ----- | ----- | ----- |
| NIQE    | 3.85  | 3.62  | 3.67  | 3.76  | 3.16  |
| BRISQUE | 24.79 | 13.01 | 12.14 | 12.52 | 19.05 |


