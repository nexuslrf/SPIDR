# Mesh-guided Scene Deformation

**With rigged mesh**

SPIDR can directly used rigged meshes for the scene deformation. This gives more flexibility for controlling the object shape, comparing against ground truth, and generating animations.

We use provided Manikin blender model as the example to show how to render the deformed scenes.

To render the we additionally need:

* object mesh and deformed mesh (they need to have the same topology/vertex order, but no need to be watertight).
* trained SPIDR model, and the extracted point cloud.

**Note:** the order of vertices of meshes should be unchanged.

In our shared dataset, we provide the deformed GT meshes with corresponding rendered images, you can use the evaluate the rendering quality of deformed scenes.

The object mesh can also be extracted from SPIDR model with

```bash

python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --marching_cube

```

then manually rig the mesh with tools like Blender.

To get the point cloud, you can use the provided script:

```bash

python ckpt2pcd.py --save_dir ../checkpoints/nerfsynth_sdf/manikin --ckpt 120000_net_ray_marching.pth --pcd 120000_pcd.ply

```


To transfer the deformation from mesh to point cloud, use the following script:

```bash

python mesh2pcd.py --mesh frame_00.obj --mesh_deformed frame_01.obj --pcd 120000_pcd.ply --pcd_deformed 120000_pcd_deformed.ply

```

Then write the deformed point cloud to the SPIDR model:

```bash

python pcd2ckpt.py --save_dir ../checkpoints/nerfsynth_sdf/manikin --ckpt 120000_net_ray_marching.pth --ckpt_deformed 120010_net_ray_marching.pth --pcd 120000_pcd_deformed.ply

```

Note we simply use `120010` to indicate the deformed model, you can use any number you want.

Finally, render the deformed scene:

```bash

python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --split=test --resume_iter 120010

```

For the rendering with BDRF estimations, we need to first re-bake the depth maps from the light sources (since the object geometry has changed).

```bash

python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --bake_light --down_sample=0.5 --resume_iter 120010

```

Then, with the baked light depth maps, we can run the PBR rendering branch with `run_mode=lighting`:

```bash

python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=lighting --split=test --resume_iter 120010

```
