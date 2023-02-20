{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Mesh-guided Scene Deformation \n",
    "\n",
    "**With rigged mesh**\n",
    "\n",
    "SPIDR can directly used rigged meshes for the scene deformation. This gives more flexibility for controlling the object shape, comparing against ground truth, and generating animations.\n",
    "\n",
    "We use provided Manikin blender model as the example to show how to render the deformed scenes.\n",
    "\n",
    "To render the we additionally need:\n",
    "* object mesh and deformed mesh (they need to have the same topology/vertex order, but no need to be watertight).\n",
    "* trained SPIDR model, and the extracted point cloud.\n",
    "\n",
    "**Note:** the order of vertices of meshes should be unchanged.\n",
    "\n",
    "In our shared dataset, we provide the deformed GT meshes with corresponding rendered images, you can use the evaluate the rendering quality of deformed scenes.\n",
    "\n",
    "The object mesh can also be extracted from SPIDR model with\n",
    "```bash\n",
    "python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --marching_cube\n",
    "```\n",
    "then manually rig the mesh with tools like Blender.\n",
    "\n",
    "\n",
    "To get the point cloud, you can use the provided script:\n",
    "```bash\n",
    "python ckpt2pcd.py --save_dir ../checkpoints/nerfsynth_sdf/manikin --ckpt 120000_net_ray_marching.pth --pcd_file 120000_pcd.ply\n",
    "```"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To transfer the deformation from mesh to point cloud, use the following script:\n",
    "```bash\n",
    "python mesh2pcd.py --mesh frame_00.obj --mesh_deformed frame_01.obj --pcd 120000_pcd.ply --pcd_deformed 120000_pcd_deformed.ply\n",
    "```\n",
    "\n",
    "Then write the deformed point cloud to the SPIDR model:\n",
    "```bash\n",
    "python pcd2ckpt.py --save_dir ../checkpoints/nerfsynth_sdf/manikin --ckpt 120000_net_ray_marching.pth --ckpt_deformed 120010_net_ray_marching.pth --pcd 120000_pcd_deformed.ply\n",
    "```\n",
    "Note we simply use `120010` to indicate the deformed model, you can use any number you want.\n",
    "\n",
    "Finally, render the deformed scene:\n",
    "```bash\n",
    "python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --split=test --resume_iter 120010\n",
    "```\n",
    "\n",
    "For the rendering with BDRF estimations, we need to first re-bake the depth maps from the light sources (since the object geometry has changed).\n",
    "```bash\n",
    "python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --bake_light --down_sample=0.5\n",
    "```\n",
    "\n",
    "Then, with the baked light depth maps, we can run the BRDF-based rendering branch.\n",
    "```bash\n",
    "python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=sdf --bake_light --down_sample=0.5 --resume_iter 120010\n",
    "```\n",
    "Then render with `run_mode=lighting`:\n",
    "```bash\n",
    "python test_ft.py --config ../dev_scripts/spidr/manikin.ini --run_mode=lighting --split=test --resume_iter 120010\n",
    "```\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "pytorch1.8",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.7.10"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "2ad409c7969d0b68622c5d3fd801faf21d5177fa8c0d44e0c25c1115acc4f9a1"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
