import numpy as np

def oneHotEncode(initial_array):
    """
    One-hot encode the labels for U-Net model
    """
    allowed_max_class_num = 2
    output_shape = list(initial_array.shape)
    output_shape[-1] = allowed_max_class_num
    output_array_dims = list(initial_array.shape)
    output_array_dims.append(allowed_max_class_num)
    output_array = np.zeros(output_array_dims)
    for image_i in range(0, initial_array.shape[0]):
        for class_num in range(0, allowed_max_class_num):
            for x in range(0, initial_array.shape[1]):
                for y in range(0, initial_array.shape[2]):
                    if initial_array[image_i, x, y] == class_num:
                        output_array[image_i, x, y, class_num] = 1
    return output_array

def makeXbyY(data, X, Y):
    """
    Crop the input data to the specified size

    Args:
    data: A 4D numpy array of shape (num_samples, x, y, channels)
    X: The desired height of the cropped images
    Y: The desired width of the cropped images

    Returns:
    A 4D numpy array of shape (num_samples, X, Y, channels) representing the cropped data
    """
    if len(data.shape) < 3:
        print('Input should be of size (num_samples, x, y, ...)')
        return None
    x_length, y_length = data.shape[1], data.shape[2]
    data_x_start = int((x_length - X) / 2)
    data_y_start = int((y_length - Y) / 2)
    arrayXbyY = data[:, data_x_start:(data_x_start + X), data_y_start:(data_y_start + Y), ...]
    return arrayXbyY
