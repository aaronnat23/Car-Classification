# Car-Classification
classifying cars

## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- [Scipy](https://www.scipy.org/)

# Pre-Processing Image
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
  --image_dir=./train
```
- to train the dataset

# Test Dataset
```bash
python classify.py path/to/image 
```
- to test single image 
