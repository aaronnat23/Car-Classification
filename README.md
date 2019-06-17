## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- [Scipy](https://www.scipy.org/)

## Dataset
We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

```bash
$ cd Car-Recognition
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

# Pre-Processing Image
-add cars_train, cars_test and car_devkit files to this folder

```bash
 python pre-process.py 
```
- to split train dataset by train/valid/test

# Train Dataset
```bash
python train.py \
  --bottleneck_dir=logs/bottlenecks \
  --how_many_training_steps=100000 \
  --model_dir=inception \
  --summaries_dir=logs/training_summaries/basic \
  --output_graph=logs/trained_graph.pb \
  --output_labels=logs/trained_labels.txt \
  --image_dir=./data/train
```
- to train the dataset

# Test Dataset
download trained model files here and save to **logs** folder of the main directory
https://drive.google.com/drive/folders/1UJFZ4DdFAVerVFvLWqbtFE7tMInd51Ir?usp=sharing
```bash
python classify.py path/to/image 
```
- to test single image 

#### Test acc:
**59.64**

