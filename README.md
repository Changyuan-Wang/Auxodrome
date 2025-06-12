## The Fruit Fly Auxodrome: a computer vision setup for longitudinal studies of Drosophila development 

<img src="https://github.com/Changyuan-Wang/Auxodrome/blob/main/auxodrome_name.png" width="400">

---

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
   - In the [pytorch3dunet/datasets](https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/datasets) folder, replace [hdf5.py](https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/datasets/hdf5.py) with the version provided in the [substitution folder](https://github.com/Changyuan-Wang/Auxodrome/tree/main/Substitution) of this repository.
   
   - In the [pytorch3dunet/unet3d](https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d) folder, replace [predictor.py](https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/predictor.py) with the version provided in the same folder.
  

---

### Training & Validation

1. Install [QuPath](https://qupath.github.io/).


2. Annotate larvae as foreground and all other components (e.g., food, eggs, pupae, etc.) as background.


3. Export annotations using the provided [export script](https://github.com/Changyuan-Wang/Auxodrome/blob/main/export_annotations.groovy). Label indices should be: foreground = 1, background = 0, unlabeled = 2 (ignored during training).


4. Convert the training and validation datasets into HDF5 format with /raw and /label datasets, as specified by [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet). A minimum of 16 frames along the time axis is required for both training and validation sets.


5. An example training YAML file is provided as train_config.yml in the [example folder](https://github.com/Changyuan-Wang/Auxodrome/tree/main/Example).

---

### YAML file generator

1. We use a [YAML file generator](https://github.com/Changyuan-Wang/Auxodrome/blob/main/YAMLfileGenerator.ipynb) to create YAML files for analyzing experimental videos. This generator identifies the center of each well, and creates a testing yaml file for each well separately. Use the trained model for each larval stage (eggs & L1, L2, L3 & pupae) and adults to run tests on that stage separately.
   
2. There are two types of testing yaml files, test_config-VideoDataset.yml and test_config-ProbField.yml. The example yaml files for one example well in the [example folder](https://github.com/Changyuan-Wang/Auxodrome/tree/main/Example).
   
    - The VideoDataset yaml file is to use the trained 3D-Unet model to run predictions on the testing frames you specified, and it will generate batches of raw frames and predicted frames for your specified well. The raw frames are just original avi videos of that well, and the predicted frames are a probability field representing the probability of each pixel being the foreground.
   
    - Use [CombineVideo.ipynb](https://github.com/Changyuan-Wang/Auxodrome/blob/main/CombineVideo.ipynb) to combine all the batches of predicted frames into a large video for each well. Then use ProbField yaml file to threshold the probability field, turn the thresholded predicted frames into batches of avi videos, calculate the areas and centroids of the predicted larvae and save those two metrics into batches csv files for future analysis.


---

### Hatch, Pupation, and Eclosion Analysis

We use a [PlotGenerator](https://github.com/Changyuan-Wang/Auxodrome/blob/main/PlotGenerator.ipynb) to find the timings of hatching, pupation, and eclosion for all wells. This ipynb file will read the csv files generated from ProbField testings, apply noise filters on the generated metrics, and spit out the timings of hatching, pupation, and eclosion.





