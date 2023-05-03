import h5py
import numpy as np
from preprocessing import oneHotEncode, label012Chromosomes, makeXbyY

from Model import U_net

# Load the data from the file
with h5py.File('LowRes_13434_overlapping_pairs.h5', 'r') as hf:
    images = hf['dataset_1'][:]
    labels = hf['dataset_1'][:]

# Normalize the images
images = images / 255.0

# Preprocess the labels
labels = label012Chromosomes(labels)

# One-hot encode the labels
one_hot_labels = oneHotEncode(labels)

# Crop the data to the required size
input_shape = (512, 512, 1)  # Change this to match the input_shape of your U-Net model
cropped_images = makeXbyY(images, input_shape[0], input_shape[1])
cropped_labels = makeXbyY(one_hot_labels, input_shape[0], input_shape[1])

# Load the pre-trained model
model = U_net(input_shape=(512, 512, 3), num_classes=2)
model.load_weights('unet_weights.h5')

# Predict using the model
predictions = model.predict(cropped_images)

# Save the preprocessed data and predictions to a new file
with h5py.File('preprocessed_data.h5', 'w') as hf:
    hf.create_dataset('input', data=cropped_images)
    hf.create_dataset('output', data=cropped_labels)
    hf.create_dataset('predictions', data=predictions)

