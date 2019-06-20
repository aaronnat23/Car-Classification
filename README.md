## Stanford Car Classification
A image classification program that uses Google's Machine Learning library, Tensorflow and a pre-trained Deep Learning Convolutional Neural Network model called Inception V3.

This model has been pre-trained for the ImageNet Large Visual Recognition Challenge using the data from 2012, and it can differentiate between 1,000 different classes, like Dalmatian, dishwasher etc. The program applies Transfer Learning to this existing model and re-trains it to classify a new set of images.

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
- To split train dataset by train/valid/test
- We augmented the color of each images to generate 2 more images. 
- This is done to increase the number of train dataset for each class.
- Each original image was converted to from RGB to BGR and BGR to HSV respectively.

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
- the inception model file will be dowloaded once the code runs
- images are augmented in different distortions like crops, scales, and flips.

# Test Dataset
download trained model files here and save to **logs** folder of the main directory
https://drive.google.com/drive/folders/1UJFZ4DdFAVerVFvLWqbtFE7tMInd51Ir?usp=sharing
```bash
python classify.py path/to/image 
```
- to test single image 

#### Test acc:
**62.34%**
- The more complex model (ex. InceptionV3) the less accurate results are. This is understanable due to bias/variance problem.

# Results
![Alt text](data/00020.jpg?raw=true "Title")
BMW X3 SUV 2012 (score = 0.60578)
Bentley Continental Flying Spur Sedan 2007 (score = 0.02981)
BMW X5 SUV 2007 (score = 0.02563)
Mercedes-Benz E-Class Sedan 2012 (score = 0.02369)
Audi V8 Sedan 1994 (score = 0.02163)
BMW 3 Series Sedan 2012 (score = 0.01933)
Audi S6 Sedan 2011 (score = 0.01916)
BMW ActiveHybrid 5 Sedan 2012 (score = 0.01839)
Audi 100 Wagon 1994 (score = 0.01519)
BMW M5 Sedan 2010 (score = 0.01504)
