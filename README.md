# Argus-3D: Pushing the Limits of 3D Shape Generation at Scale
[**Paper**](https://arxiv.org/abs/2306.11510) | [**Project Page**](https://argus-3d.github.io)

## Installation
You can create an anaconda environment called `argus-3d` using
```
conda env create -f environment.yaml
conda activate argus-3d
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

## Generation
Download [stage1 checkpoint](https://drive.google.com/file/d/12_H2AzXIE8F1dEwfJbaezJmcqb7urERw/view?usp=sharing) and place it into `output/PR256_ED512_EN8192`.

Download [stage2 checkpoint](https://drive.google.com/file/d/10lRH2XMOEwpsr2Ho_rxtRT6MMybLyeD-/view?usp=sharing) and place it into `output/PR256_ED512_EN8192/class-guide/transformer3072_24_32`.

Then you can try class-guide generation by run:
```
python generate_class-guide.py --batch_size 16 --cate chair
```
This script should create a folder `output/PR256_ED512_EN8192/class-guide/transformer3072_24_32/class_cond` where the output meshes are stored.

**Note**: Our model requires significant memory, and it's recommended to run it on a GPU with high VRAM capacity (40GB or above). Generating a single mesh on the A100 (80GB) takes approximately 50 seconds on average, while on V100 (32GB) it takes ~6 minutes.

## Dataset
The occupancies, point clouds, and supplementary rendered images based on the Objaverse dataset can be downloaded from [https://huggingface.co/datasets/BAAI/Objaverse-MIX](https://huggingface.co/datasets/BAAI/Objaverse-MIX)

### Coming Soon

* Image-guide generation
* Text-guide generation
* Training code

## Shout-outs
Thanks to everyone who makes their code and models available.
- [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks)
- [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers)

Thanks for open-sourcing!

## BibTeX

```
@misc{yu2023pushing,
      title={Pushing the Limits of 3D Shape Generation at Scale}, 
      author={Yu Wang and Xuelin Qian and Jingyang Huo and Tiejun Huang and Bo Zhao and Yanwei Fu},
      year={2023},
      eprint={2306.11510},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
