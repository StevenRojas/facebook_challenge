# Flower Classifier
Train an Image Classifier to Identify Flowers (Udacity & Facebook Challenge)

## Setup
- Download [the images](https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip) to train and classify and put them at data/flowers folder
- Make sure you have Python 3 environment, preferred to use [Anaconda](https://www.anaconda.com/download/)

## Usage
### Training network
- Run train.py script with the following parameters:
```buildoutcfg
train.py [-h] [--save_chk SAVE_CHK] [--arch ARCH] [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS] 
    [--epochs EPOCHS] [--gpu GPU] [--config CONFIG] data_dir
```

    data_dir: Path to images folder
    --save_chk: Name of the checkpoint file

- Trained checkpoint will be saved at data/checkpoints folder

### Network architecture finder
- TBD

### Predicting images
- TBD
