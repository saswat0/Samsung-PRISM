# AdaMatting
"Disentangled Image Matting"  for Samsung PRISM Project

## Dependencies
+ python 3.6  
+ pytorch 1.4.0  
+ tensorboardX  
+ opencv  

## Datasets
### Adobe Composition-1k Dataset
[Contact author](https://sites.google.com/view/deepimagematting) for the dataset.  
### MSCOCO
[train2014.zip](http://images.cocodataset.org/zips/train2014.zip)  
### PASCAL VOC
[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/):  
+ [VOCtrainval_14-Jul-2008.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar)  
+ [VOC2008test.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html)  

## Usage
Download and move all above compressed data files into a folder.  
Set `--raw_data_path={path_to_your_folder}` in any `.sh` file you use. 
### Decompress and composite the data
```bash
$ bash ./prep.sh
```
### Train
```bash
$ bash ./train.sh
```
### Visualize training
```bash
$ tensorboard --logdir ./runs
```

### Current Performance and logs
Trained on 60 epochs
SAD - 88.0
MSE - 0.013 on Adobe-1k Composition dataset