# MXNet ResNet

Train ResNet on Imagenet 1000 dataset.

## Setup

### Download Imagenet Dataset

Download ImageNet LSVRC 2012 Object Classification/Detection.
* Training Images (Task 1 & 2). ILSVRC2012_img_train.tar (138GB)
* Validation Images (All tasks). ILSVRC_img_val.tar (6.3GB)

Download Imagent ILSVRC 2015 Development kit.
* ILSVRC2015_devkit.tar.gz

```bash
# Create dataset directory
mkdir -p datasets/imagenet/ILSVRC2015
mkdir -p datasets/imagenet/lists
mkdir -p datasets/imagenet/rec

# Extract devkit
tar -xvzf ILSVRC2015_devkit.tar.gz
mv ILSVRC2015/devkit datasets/imagenet/ILSVRC2015/devkit

# Extract ImageSets
git clone https://github.com/wcl20/MXNet-ResNet.git
mv MXNet-GoogLeNet/ImageSets datasets/imagenet/ILSVRC2015/ImageSets

# Extract training/validation data
mkdir -p datasets/imagenet/ILSVRC2015/Data/CLS-LOC/train
mkdir -p datasets/imagenet/ILSVRC2015/Data/CLS-LOC/val
mv ILSVRC2012_img_train.tar datasets/imagenet/ILSVRC2015/Data/CLS-LOC/train
mv ILSVRC2012_img_val.tar datasets/imagenet/ILSVRC2015/Data/CLS-LOC/val
cd datasets/imagenet/ILSVRC2015/Data/CLS-LOC/train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ../val
tar -xvf ILSVRC2012_img_val.tar
```

### Build mxnet lists
```bash
python3 build.py
```

### Build mxnet image records
MXnet provides python script to create image records.
```bash
git clone https://github.com/apache/incubator-mxnet.git
python3 $MXNET_HOME/tools/im2rec.py <path to train.lst> "" --resize 256 --encoding .jpg --quality 100
python3 $MXNET_HOME/tools/im2rec.py <path to val.lst> "" --resize 256 --encoding .jpg --quality 100
python3 $MXNET_HOME/tools/im2rec.py <path to test.lst> "" --resize 256 --encoding .jpg --quality 100
mv datasets/imagenet/lists/*.rec datasets/imagenet/rec/
mv datasets/imagenet/lists/*.idx datasets/imagenet/rec/
```

### Final Repository
    .
    ├── datasets
    │   ├── imagenet
    │   │   ├── ILSVRC2015
    │   │   │   ├── Data  
    │   │   │   │   └── CLS-LOC
    │   │   │   │   │   ├── train
    │   │   │   │   │   │   ├── n01440764
    │   │   │   │   │   │   │   ├── n01440764_10026.JPEG
    │   │   │   │   │   │   │   ├── n01440764_10027.JPEG
    │   │   │   │   │   │   │   ├── ...
    │   │   │   │   │   │   ├── ...
    │   │   │   │   │   └── val
    │   │   │   │   │   │   ├── ILSVRC2012_val_00000001.JPEG
    │   │   │   │   │   │   ├── ILSVRC2012_val_00000002.JPEG
    │   │   │   │   │   │   ├── ...
    │   │   │   ├── devkit  
    │   │   │   │   ├── data
    │   │   │   │   │   ├── ILSVRC2015_clsloc_validation_blacklist.txt
    │   │   │   │   │   ├── ILSVRC2015_clsloc_validation_ground_truth.txt
    │   │   │   │   │   ├── ...
    │   │   │   │   ├── evaluation
    │   │   │   │   ├── COPYING
    │   │   │   │   └── readme.txt
    │   │   │   └── ImageSets  
    │   │   │   │   └── CLS-LOC
    │   │   │   │   │   ├── train_cls.txt
    │   │   │   │   │   └── val.txt
    │   │   ├── lists
    │   │   │   ├── test.lst
    │   │   │   ├── train.lst
    │   │   │   └── val.lst
    │   │   └── rec
    │   │   │   ├── test.idx
    │   │   │   ├── test.rec
    │   │   │   ├── train.idx
    │   │   │   ├── train.rec
    │   │   │   ├── val.idx
    │   │   │   └── val.rec
    ├── incubator-mxnet
    │   ├── tools
    │   │   ├── im2rec.py
    │   │   ├── ...
    │   ├── ...
    ├── MXNet-ResNet          
    │   ├── config          
    │   ├── core  
    │   ├── output
    │   │   └── mean.json
    │   ├── build.py
    │   ├── train.py
    │   └── ...

## Train
```bash
cd MXNet-ResNet
python3 train.py --checkpoints checkpoints --prefix resnet
```

## Test
```bash
python3 test.py --checkpoints checkpoints --prefix resnet
```
