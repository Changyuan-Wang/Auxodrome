## The Fruit Fly Auxodrome: a computer vision setup for longitudinal studies of Drosophila development 
**Auxo-**: growth, increase; a Greek goddess representing growth  
**-drome**: a place for running or racing  

“**Auxodrome**” is also a real word referring to “a plotted curve indicating the relative development of a child at any given age.”

---

### Installation & Substitution  

Please follow the installation instructions provided on the [pytorch-3dunet GitHub page](https://github.com/wolny/pytorch-3dunet).

After completing those steps:

1. Create a conda environment as instructed on the [pytorch-3dunet page](https://github.com/wolny/pytorch-3dunet) and activate it.  
2. Install the required packages listed on the [pytorch-3dunet page](https://github.com/wolny/pytorch-3dunet), and additionally:  
   ```
   conda install -c pytorch torchvision pytorch -c conda-forge numpy
   ```
3. Replace the following files in the cloned pytorch-3dunet repository:
   (1) In [pytorch3dunet/datasets](https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/datasets), replace [hdf5.py](https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/datasets/hdf5.py) with the version provided in the substitution folder of this repository.
   (2) In [pytorch3dunet/unet3d](https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d), replace [predictor.py](https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/predictor.py) with the version provided in the same folder.




