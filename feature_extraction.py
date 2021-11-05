from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

conv_base = None # TODO: Student - Load the VGG16 model (see writeup)
datagen = None   # TODO: Student - create a data generator  (see part 3)
batch_size = 0   # TODO: Student define a batch size (see writeup)

base_dir = 'cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


def extract_features(directory, sample_count):
    """
    Given a directory of images, return the features from the VGG16 model, along with 
    the corresponding cat / dog labels.

    This function is almost finished for you.
    TODO: Student - add the call to flow_from_directory like you did before
    """
    # Initialize features and labels vectors
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))

    # Initialize a generator (call flow_from_directory)
    generator = None # TODO: Student: call flow_from_directory as in Part 3
        
    # Iterate over the generator and extract features and labels
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# TODO: Student - call extract_features() for each directory to retrieve the features and labels for the images in that directory
train_features, train_labels = None

# TODO Student - flatten the features for use in a fully connected network network (for each directory)
# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))

# TODO: Student - save each to a file to load in train_feature_extraction.py
# np.save("results/train_features.npy", train_features)