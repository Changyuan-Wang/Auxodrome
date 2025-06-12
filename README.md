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
   - In the [pytorch3dunet/datasets](https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/datasets) folder, replace [hdf5.py](https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/datasets/hdf5.py) with the version provided in the substitution folder of this repository.
   - In the [pytorch3dunet/unet3d](https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d) folder, replace [predictor.py](https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/predictor.py) with the version provided in the same folder.
  

---

### Training & Validation

1. Install [QuPath](https://qupath.github.io/).


2. Annotate larvae as foreground and all other components (e.g., food, eggs, pupae, etc.) as background.


3. Export annotations using the provided export script. Label indices should be: foreground = 1, background = 0, unlabeled = 2 (ignored during training).


4. Convert the training and validation datasets into HDF5 format with /raw and /label datasets, as specified by [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet). A minimum of 16 frames along the time axis is required for both training and validation sets.


5. An example training YAML file is provided as train_config.yml in the example folder.




