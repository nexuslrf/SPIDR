# SPIDR

[Project Page](https://nexuslrf.github.io/SPIDR_webpage/) | [Paper](https://arxiv.org/abs/2210.08398)

![demo](https://nexuslrf.github.io/images/lego_demo.gif)

Codes for: "SPIDR: SDF-based Neural Point Fields for Illumination and Deformation"

**UPDATE**: I am still actively writing/testing instruction and demo examples.

## Install

```bash
git clone https://github.com/nexuslrf/SPIDR.gi
cd SPIDR
```

**Environment**

```bash
pip install -r requirements.txt
```

**Torch extensions**

We replaced original [PointNeRF](https://github.com/Xharlie/pointnerf)'s pycuda kernels with torch extensions. To set up our torch extensions for ray marching:

```bash
cd models/neural_points/c_ext
python setup.py build_ext --inplace
```

We have tested our codes on torch 1.8, 1.10, 1.11.

## Data Prepare

Download the dataset from the following links and put them under `./data_src/` directory:

[NeRF-synthetic]([https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)) scenes (`./data_src/nerf_synthetic`)

[NVSF-BlendedMVS](https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip) scenes (`./data_src/BlendedMVS`)

Our added [scenes for deformation](https://drive.google.com/drive/folders/1zlHdPJST47psbEbrC71PWl04kfjH2GHe?usp=sharing) (`./data_src/deform_synthetic`)

We also provided some pre-trained examples: 

## Usage

### Training

We'll update instructions later

**First stage**: train a Point-based NeRF model, this step is similar to the original PointNeRF.

**Second stage**: train the BRDF + Environment light MLPs

The second stage of the training requires pre-computing the depth maps from the light sources

```bash
python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --bake_light --down_sample=0.5
```

`--down_sample=0.5` halve the size of the rendered depth images.

### Testing

We use manikin scene as an example.

To simply render frames (SPIDR* in the paper):

```bash
cd run/
python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --split=test
```

For the rendering with BDRF estimations.

We need to first bake the depth maps from the light sources. If you did it during the training BDRF, you don't need to run it again (but it requires updates if the object shape is changed).

```bash
python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --bake_light --down_sample=0.5
```

Then, with the baked light depth maps, we can run the BRDF-based rendering branch.

```bash
python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=lighting --split=test
```

### Editing

##### Extracting Mesh

```bash
python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --marching_cube
```

##### Extracting Point Cloud

```bash
cd ../deform_tools
python ckpt2pcd.py --save_dir ../checkpoints/nerfsynth_sdf/manikin --ckpt 120000_net_ray_marching.pth --pcd_file 120000_pcd.ply
```

#### Deforming

We provide three examples of different editing:

* Deformation with GT deformed meshes
* Deformation with the extracted mesh from the our model
* Direct point manipulations

#### Relighting

Simply add target environment HDRI in `--light_env_path`

```bash
python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=lighting --split=test --light_env_path=XXX.hdr
```

Note: 

* the HDRI should be resized to `32x16` resolution before the relighting.
* light intensity can be scaled by flag `--light_intensity` e.g., `--light_intensity=1.7`

![demo_relight](https://nexuslrf.github.io/images/vid.gif)

## Citation

```
@article{liang2022spidr,
  title={SPIDR: SDF-based Neural Point Fields for Illumination and Deformation},
  author={Liang, Ruofan and Zhang, Jiahao and Li, Haoda and Yang, Chen and Guan, Yushi and Vijaykumar, Nandita},
  journal={arXiv preprint arXiv:2210.08398},
  year={2022}
}
```
