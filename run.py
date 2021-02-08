"""
Test GPU
Author: Swapnil Masurekar
"""
import os
import argparse

import tensorflow as tf
import pandas as pd

tf.get_logger().setLevel('INFO')


def get_args():
    """Get required args"""
    args_parser = argparse.ArgumentParser()
    # Experiment arguments
    args_parser.add_argument(
      '--batch-size',
      required=False,
      default=256,
      type=int
    )
    args_parser.add_argument(
      '--train-steps',
      required=False,
      default=5000,
      type=int
    )
    args_parser.add_argument(
      '--device',
      required=True,
      type=str
    )
    return args_parser.parse_args()

args = get_args()

BATCH_SIZE = args.batch_size
TRAIN_STEPS = args.train_steps
DEVICE = args.device.lower()

assert DEVICE in ['cpu', 'gpu'], "--device must is in ['cpu', 'gpu']"

# Select device
if DEVICE == 'cpu':
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# TEST CODE ------

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

print('-'*30)
print("Loading Data ...")

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')



def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
print('-'*30)
print("Building Model ...")
    
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

print('-'*100)
print("Training Model on", DEVICE,"...")



# Train the Model.
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True, batch_size=BATCH_SIZE),
    steps=TRAIN_STEPS)
