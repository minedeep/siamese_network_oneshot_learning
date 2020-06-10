import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


class ImageAugmentor:

    """
    rotation_range : degrees (0 to 180)
    augmentation_probability: probablity of augmentation
    """

    def __init__(self, augmentation_probability, rotation_range):

        self.augmentation_probability = augmentation_probability
        self.rotation_range = rotation_range

    def _transform_matrix_offset_center(self, transformation_matrix, width, height):
        """
        Corrects the offset of transformation matrix
        """

        ox = float(width)/2 + 0.5
        oy = float(height)/2 + 0.5

        offset_matrix = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]])

        transformation_matrix = np.dot(np.dot(offset_matrix, transformation_matrix), reset_matrix)
        return transformation_matrix

    def _apply_transform(self, image, transformation_matrix):

        """
        Apply the rpovided transformation to the image
        """

        channel_axis = 2
        image = np.rollaxis(image, channel_axis, 0)
        final_affine_matrix = transformation_matrix[:2,:2]
        final_offset = transformation_matrix[:2, 2]

        channel_images = [ndi.interpolation.affine_transform(
            image_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode='nearest',
            cval=0) for image_channel in image]

        image = np.stack(channel_images, axis=0)
        image=np.rollaxis(image, 0, channel_axis+1)

        return image

    def perform_random_rotation(self, image):

        """
        Apply a random rotation
        """

        random_number = np.random.random()
        if random_number <= self.transform_probability:
            return image
        theta = np.deg2rad(np.random.uniform(low = self.rotation_range[0],
                                             high = self.rotation_range[1]))

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0]
                                    [np.sin(theta), np.cos(theta), 0]
                                    [0, 0, 1]])

        transformation_matrix = self._transform_matrix_offset_center(
                            rotation_matrix, image.shape[0], image.shape[1])

        image = self._apply_transform(image, transformation_matrix)

        return image

