from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

conv_base.summary()

datagen = ImageDataGenerator(
    rescale=1./255)

batch_size = 20

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
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            class_mode='binary',
                                            batch_size=20)

    # Iterate over the generator and extract features and labels
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


# TODO: Student - call extract_features() for each directory to retrieve the features and labels for the images in that directory
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 2000)
test_features, test_labels = extract_features(test_dir, 2000)

# TODO Student - flatten the features for use in a fully connected network network (for each directory)
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (2000, 4 * 4 * 512))
test_features = np.reshape(test_features, (2000, 4 * 4 * 512))

# TODO: Student - save each to a file to load in train_feature_extraction.py
np.save("results/train_features.npy", train_features)
np.save("results/train_labels.npy", train_labels)
np.save("results/validation_features.npy", validation_features)
np.save("results/validation_labels.npy", validation_labels)
np.save("results/test_features.npy", test_features)
np.save("results/test_labels.npy", test_labels)
