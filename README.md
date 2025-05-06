# **NiRNE**
This repository contains NiRNE, the image-to-normal estimator of [Hi3DGen](https://github.com/Stable-X/Hi3DGen)

## News
- Release [NiRNE](https://github.com/lzt02/NiRNE/) :fire::fire::fire: (05.06, 2025 UTC)

## Installation:
Please run following commands to build package:
```
git clone https://github.com/lzt02/NiRNE.git
cd NiRNE
pip install -r requirements.txt
```

## Start Quickly
```
python infer.py --input_dir data --output_dir output
```

## Usage
To use the StableNormal pipeline, you can instantiate the model and apply it to an image as follows:

```python
import torch
from PIL import Image

# Load an image
input_image = Image.open("path/to/your/image.jpg")

# Create predictor instance
predictor = torch.hub.load("lzt02/NiRNE", "NiRNE", trust_repo=True)

# Apply the model to the image
normal_image = predictor(input_image)

# Save or display the result
normal_image.save("output/normal_map.png")
```
- If Hugging Face is not available from terminal, you could download the pretrained weights to `weights` dir:

```python
predictor = torch.hub.load("lzt02/NiRNE", "NiRNE", trust_repo=True, local_cache_dir='./weights')
```

## Citation
If you find this work helpful, please consider citing our paper:
```bibtex
@article{ye2025hi3dgen,
  title={Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging},
  author={Ye, Chongjie and Wu, Yushuang and Lu, Ziteng and Chang, Jiahao and Guo, Xiaoyang and Zhou, Jiaqing and Zhao, Hao and Han, Xiaoguang},
  journal={arXiv preprint arXiv:2503.22236}, 
  year={2025}
}
```
