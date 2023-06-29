# Argus-3D
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
You can try class-guide generation by run:
```
python generate_class-guide.py --batch_size=16 --cate chair
```
This script should create a folder `output/PR256_ED512_EN8192/class-guide/transformer3072_24_32/class_cond` where the output meshes are stored.

**Note**: Our model requires a minimum of 18GB of graphics memory. Generating a single mesh on the A100 (80GB) takes approximately 40 seconds.

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
      author={Wang Yu and Xuelin Qian and Jingyang Huo and Tiejun Huang and Bo Zhao and Yanwei Fu},
      year={2023},
      eprint={2306.11510},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
