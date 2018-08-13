"""
image_generators.py

Image generators for training convolutional neural networks

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from fnmatch import fnmatch

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel_h
from skimage.filters import sobel_v
from skimage.measure import label
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.io import imread

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical as keras_to_categorical
from tensorflow.python.keras.preprocessing.image import random_channel_shift
from tensorflow.python.keras.preprocessing.image import apply_transform
from tensorflow.python.keras.preprocessing.image import flip_axis
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from keras_retinanet.preprocessing.generator import Generator as _RetinaNetGenerator
from keras_maskrcnn.preprocessing.generator import Generator as _MaskRCNNGenerator

from .utils.transform_utils import transform_matrix_offset_center
from .utils.transform_utils import distance_transform_2d, distance_transform_3d
from .utils.retinanet_anchor_utils import anchor_targets_bbox


"""
Custom image generators
"""


class ImageSampleArrayIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if train_dict['y'] is not None and train_dict['X'].shape[0] != train_dict['y'].shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 train_dict['X'].shape, train_dict['y'].shape))
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ImageSampleArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.y = train_dict['y']
        self.win_x = train_dict['win_x']
        self.win_y = train_dict['win_y']
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ImageSampleArrayIterator, self).__init__(
            len(train_dict['y']), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.channel_axis == 1:
            batch_x = np.zeros((len(index_array),
                                self.x.shape[self.channel_axis],
                                2 * self.win_x + 1,
                                2 * self.win_y + 1))
        else:
            batch_x = np.zeros((len(index_array),
                                2 * self.win_x + 1,
                                2 * self.win_y + 1,
                                self.x.shape[self.channel_axis]))

        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class SampleDataGenerator(ImageDataGenerator):
    def flow(self,
             train_dict,
             batch_size=32,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return ImageSampleArrayIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class ImageFullyConvIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 target_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ImageFullyConvIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.y = np.array(train_dict['y'], dtype='int32')
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.target_format = target_format
        super(ImageFullyConvIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        epsilon = K.epsilon()
        if self.target_format == 'direction':
            y_channel_shape = 2
        else:
            y_channel_shape = self.y.shape[self.channel_axis]

        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        if self.channel_axis == 1:
            batch_y = np.zeros(tuple([len(index_array), y_channel_shape] +
                                     list(self.y.shape)[2:]))
        else:
            batch_y = np.zeros(tuple([len(index_array)] +
                                     list(self.y.shape)[1:3] +
                                     [y_channel_shape]))

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            if self.target_format == 'direction':
                # TODO: hardcoded for cell interior = 1
                if self.channel_axis == 1:
                    interior = y[1, :, :]
                else:
                    interior = y[:, :, 1]
                distance = ndi.distance_transform_edt(interior)
                gradient_x = sobel_h(distance)
                gradient_y = sobel_v(distance)
                norm = np.sqrt(gradient_x ** 2 + gradient_y ** 2 + epsilon)
                direction_x = gradient_x / norm
                direction_y = gradient_y / norm
                direction = np.stack([direction_x, direction_y], axis=0)
                y = direction

            batch_x[i] = x
            batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img_x = np.expand_dims(batch_x[i, :, :, 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                if self.target_format == 'direction':
                    img_y = np.expand_dims(batch_y[i, :, :, 0], -1)
                    img = array_to_img(img_y, self.data_format, scale=True)
                    fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class ImageFullyConvDataGenerator(ImageDataGenerator):
    """
    Generate minibatches of image data and masks
    with real-time data augmentation.
    """

    def flow(self,
             train_dict,
             batch_size=1,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             target_format=None):
        return ImageFullyConvIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            target_format=target_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')

        return x

    def random_transform(self, x, labels=None, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 4D tensor, single image.
            labels: 4D tensor, single image mask.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.deg2rad(
                np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(
                transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([
                [1, -np.sin(shear), 0],
                [0, np.cos(shear), 0],
                [0, 0, 1]
            ])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(
                transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([
                [zx, 0, 0],
                [0, zy, 0],
                [0, 0, 1]
            ])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(
                transform_matrix, zoom_matrix)

        if labels is not None:
            y = labels  # np.expand_dims(labels, axis=0)

            if transform_matrix is not None:
                h, w = y.shape[img_row_axis], y.shape[img_col_axis]
                transform_matrix_y = transform_matrix_offset_center(transform_matrix, h, w)
                y = apply_transform(y, transform_matrix_y, img_channel_axis,
                                    fill_mode='constant', cval=0)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix_x = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix_x, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                if labels is not None:
                    y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                if labels is not None:
                    y = flip_axis(y, img_row_axis)

        if labels is not None:
            return x, y.astype('int')
        return x


"""
Custom siamese generators
"""


class SiameseDataGenerator(ImageDataGenerator):
    def flow(self,
             train_dict,
             crop_dim=32,
             min_track_length=5,
             batch_size=32,
             shuffle=True,
             seed=None,
             data_format=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return SiameseIterator(
            train_dict,
            self,
            crop_dim=crop_dim,
            min_track_length=min_track_length,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class SiameseIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 crop_dim=14,
                 min_track_length=5,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 squeeze=False,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()

        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 3
            self.col_axis = 4
            self.time_axis = 2
        if data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3
            self.time_axis = 1
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())
        self.y = np.array(train_dict['y'], dtype='int32')

        if self.x.ndim != 5:
            raise ValueError('Input data in `SiameseIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)

        self.crop_dim = crop_dim
        self.min_track_length = min_track_length
        self.image_data_generator = image_data_generator
        self.squeeze = squeeze
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        if 'daughters' in train_dict:
            self.daughters = train_dict['daughters']
        else:
            self.daughters = None

        self._remove_bad_images()
        self._create_track_ids()
        self._create_appearances()

        super(SiameseIterator, self).__init__(
            len(self.track_ids), batch_size, shuffle, seed)

    def _remove_bad_images(self):
        """
        This function goes through all of the batches of images and removes the 
        images that only have one cell.
        This could be expanded to include questionable annotations
        """
        good_batches = []
        number_of_batches = self.x.shape[0]
        for batch in range(number_of_batches):
            y = self.y[batch]
            unique_ids = np.unique(y.flatten())
            if len(unique_ids) > 2: # You should have at least 3 id's - 2 cells and 1 background
                good_batches.append(batch)

        X_new_shape = tuple([len(good_batches)] + list(self.x.shape[1:]))
        y_new_shape = tuple([len(good_batches)] + list(self.y.shape[1:]))

        X_new = np.zeros(X_new_shape, dtype = K.floatx())
        y_new = np.zeros(y_new_shape, dtype = np.int32)

        counter = 0
        for k, batch in enumerate(good_batches):
            X_new[k] = self.x[batch]
            y_new[k] = self.y[batch]

        self.x = X_new
        self.y = y_new

        return None

    def _create_track_ids(self):
        """
        This function builds the track id's. It returns a dictionary that
        contains the batch number and label number of each each track.
        Creates unique cell IDs, as cell labels are NOT unique across batches.
        """
        track_counter = 0
        track_ids = {}
        for batch in range(self.y.shape[0]):
            y_batch = self.y[batch]
            num_cells = np.amax(y_batch)
            for cell in range(1, num_cells + 1):
                # count number of pixels cell occupies in each frame
                y_true = np.sum(y_batch == cell, axis=(self.row_axis - 1, self.col_axis - 1))
                # get indices of frames where cell is present
                y_index = np.where(y_true > 0)[0]
                if y_index.size > self.min_track_length+1:  # if cell is present at all
                    if self.daughters is not None:
                        # Only include daughters if there are enough frames in their tracks
                        daughter_ids = self.daughters[batch][cell]
                        if len(daughter_ids) > 0:
                            daughter_track_lengths = []
                            for did in daughter_ids:
                                d_true = np.sum(y_batch == did, axis=(self.row_axis - 1, self.col_axis - 1))
                                d_track_length = len(np.where(d_true>0)[0])
                                daughter_track_lengths.append(d_track_length > self.min_track_length+1)
                            keep_daughters = all(daughter_track_lengths)
                            daughters = daughter_ids if keep_daughters else []
                        else:
                            daughters = []
                    else:
                        daughters = []
                            
                    track_ids[track_counter] = {
                        'batch': batch,
                        'label': cell,
                        'frames': y_index,
                        'daughters': daughters  # not [cell-1]!
                    }
                    track_counter += 1
                    
                else:
                    y_batch[y_batch == cell] = 0
                    self.y[batch] = y_batch
                    
        # Add a field to the track_ids dict that locates all of the different cells
        # in each frame

        for track in track_ids.keys():
            track_ids[track]['different'] = {}
            batch = track_ids[track]['batch']
            label = track_ids[track]['label']
            for frame in track_ids[track]['frames']:
                y_unique = np.unique(self.y[batch][frame])
                y_unique = np.delete(y_unique, np.where(y_unique == 0))
                y_unique = np.delete(y_unique, np.where(y_unique == label))
                track_ids[track]['different'][frame] = y_unique  
                        
        # We will need to look up the track_ids of cells if we know their batch and label. We will 
        # create a dictionary that stores this information
        reverse_track_ids = {}
        for batch in range(self.y.shape[0]):
            reverse_track_ids[batch] = {}
        for track in track_ids.keys():
            batch = track_ids[track]['batch']
            label = track_ids[track]['label']
            reverse_track_ids[batch][label] = track
        self.track_ids = track_ids
        self.reverse_track_ids = reverse_track_ids
        return None

    def _get_appearances(self, X, y, frames, labels):
        channel_axis = self.channel_axis - 1
        if self.data_format == 'channels_first':
            appearance_shape = (X.shape[channel_axis],
                                len(frames),
                                self.crop_dim,
                                self.crop_dim)
        else:
            appearance_shape = (len(frames),
                                self.crop_dim,
                                self.crop_dim,
                                X.shape[channel_axis])

        # Initialize storage for appearances and centroids
        appearances = np.zeros(appearance_shape, dtype=K.floatx())
        centroids = []

        for counter, (frame, cell_label) in enumerate(zip(frames, labels)):
            # Get the bounding box
            y_frame = y[frame] if self.data_format == 'channels_last' else y[:, frame]
            props = regionprops(np.int32(y_frame == cell_label))
            minr, minc, maxr, maxc = props[0].bbox
            centroids.append(props[0].centroid)

            # Extract images from bounding boxes
            if self.data_format == 'channels_first':
                appearance = X[:, frame, minr:maxr, minc:maxc]
                resize_shape = (X.shape[channel_axis], self.crop_dim, self.crop_dim)
            else:
                appearance = X[frame, minr:maxr, minc:maxc, :]
                resize_shape = (self.crop_dim, self.crop_dim, X.shape[channel_axis])

            # Resize images from bounding box
            max_value = np.amax([np.amax(appearance), np.absolute(np.amin(appearance))])
            appearance /= max_value
            appearance = resize(appearance, resize_shape)
            appearance *= max_value
            if self.data_format == 'channels_first':
                appearances[:, counter] = appearance
            else:
                appearances[counter] = appearance

        return [appearances, centroids]

    def _create_appearances(self):
        """
        This function gets the appearances of every cell, crops them out, resizes them, 
        and stores them in an matrix. Pre-fetching the appearances should significantly 
        speed up the generator
        """
        number_of_tracks = len(self.track_ids.keys())

        # Initialize the array for the appearances and centroids
        if self.data_format =='channels_first':
            all_appearances_shape = (number_of_tracks, self.x.shape[self.channel_axis], self.x.shape[self.time_axis], self.crop_dim, self.crop_dim)
        if self.data_format == 'channels_last':
            all_appearances_shape = (number_of_tracks, self.x.shape[self.time_axis], self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis])
        all_appearances = np.zeros(all_appearances_shape, dtype = K.floatx())

        all_centroids_shape = (number_of_tracks, self.x.shape[self.time_axis],2)
        all_centroids = np.zeros(all_centroids_shape, dtype = K.floatx())

        for track in self.track_ids.keys():
            batch = self.track_ids[track]['batch']
            label = self.track_ids[track]['label']
            frames = self.track_ids[track]['frames']

            # Make an array of labels that the same length as the frames array
            labels = [label] * len(frames)
            X = self.x[batch]
            y = self.y[batch]

            app = self._get_appearances(X, y, frames, labels)
            appearance = app[0]
            centroid = app[1]

            if self.data_format == 'channels_first':
                all_appearances[track,:,np.array(frames),:,:] = appearance 
            if self.data_format == 'channels_last':
                all_appearances[track,np.array(frames),:,:,:] = appearance

            all_centroids[track,np.array(frames),:] = centroid

        self.all_appearances = all_appearances
        self.all_centroids = all_centroids

        return None

    def _fetch_appearances(self, track, frames):
        """
        This function gets the appearances after they have been 
        cropped out of the image
        """
        # TO DO: Check to make sure the frames are acceptable

        if self.data_format == 'channels_first':
            appearances = self.all_appearances[track,:,np.array(frames),:,:]
        if self.data_format == 'channels_last':
            appearances = self.all_appearances[track,np.array(frames),:,:,:]
        return appearances

    def _fetch_centroids(self, track, frames):
        """
        This function gets the centroids after they have been
        extracted and stored
        """
        # TO DO: Check to make sure the frames are acceptable
        
        centroids = self.all_centroids[track,np.array(frames),:]
        return centroids

    def _fetch_frames(self, track, division=False):
        """
        This function fetches a random list of frames for a given track.
        If the division flag is true, then the list of frames ends at the cell's
        last appearance if the division flag is true.
        """
        track_id = self.track_ids[track]
        batch = track_id['batch']
        tracked_frames = list(track_id['frames'])

        # We need to have at least one future frame to pick from, so if 
        # the last frame of the movie is a tracked frame, remove it
        last_frame = self.x.shape[self.time_axis] - 1
        if last_frame in tracked_frames:
            tracked_frames.remove(last_frame)

        # Get the indices of the tracked_frames list - sometimes frames
        # are skipped
        tracked_frames_index = np.arange(len(tracked_frames))
        
        # Check if there are enough frames
        enough_frames = len(tracked_frames_index) > self.min_track_length + 1

        # We need to exclude the last frame so that we will always be able to make a comparison
        acceptable_indices = tracked_frames_index[self.min_track_length-1:-1] if enough_frames else tracked_frames_index[:-1]

        # Take the last frame if there is a division, otherwise randomly pick a frame
#         print(tracked_frames_index)
#         print(division)
#         print(acceptable_indices)
        index = -1 if division else np.random.choice(acceptable_indices) 

        # Select the frames. If there aren't enough frames, repeat the first frame
        # the necessary number of times
        if enough_frames:
            frames = tracked_frames[index+1-self.min_track_length:index+1]
        else:
            frames_temp = tracked_frames[0:index+1]
            missing_frames = self.min_track_length - len(frames_temp)
            frames = tracked_frames[0] * missing_frames + frames_temp

        return frames

    def _get_batches_of_transformed_samples(self, index_array):
        # Initialize batch_x_1, batch_x_2, and batch_y, as well as cell distance data
        # DVV Notes - I'm changing how this works. We will now only compare cells in neighboring
        # frames. I am also modifying it so it will select a sequence of cells/distances for x1
        # and 1 cell/distance for x2

        if self.data_format == 'channels_first':
            x1_shape = (len(index_array), self.x.shape[self.channel_axis], self.min_track_length, self.crop_dim, self.crop_dim)
            x2_shape = (len(index_array), self.x.shape[self.channel_axis], self.crop_dim, self.crop_dim)
        else:
            x1_shape = (len(index_array), self.min_track_length, self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis])
            x2_shape = (len(index_array), 1, self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis])
        distance_shape_1 = (len(index_array), self.min_track_length, 2)
        distance_shape_2 = (len(index_array), 1, 2)
        y_shape = (len(index_array), 3)

        batch_x_1 = np.zeros(x1_shape, dtype=K.floatx())
        batch_x_2 = np.zeros(x2_shape, dtype=K.floatx())
        batch_distance_1 = np.zeros(distance_shape_1, dtype=K.floatx())
        batch_distance_2 = np.zeros(distance_shape_2, dtype=K.floatx())
        batch_y = np.zeros(y_shape, dtype=np.int32)

        for i, j in enumerate(index_array):
            # Identify which tracks are going to be selected
            track_id = self.track_ids[j]
            batch = track_id['batch']
            label_1 = track_id['label']  
            
            X = self.x[batch]
            y = self.y[batch]

            # Choose comparison cell
            # Determine what class the track will be - different (0), same (1), division (2)
            # is_same_cell = np.random.random_integers(0, 1)
            division = False
            type_cell = np.random.randint(0, 3)

            # Dealing with edge cases

            # If class is division, check if the first cell divides. If not, change class to same/dif randomly
            if type_cell == 2:
                daughters = track_id['daughters']
                if len(daughters) == 0:
                    type_cell = np.random.random_integers(0, 1)  # No children so randomly choose a diff class
                else:
                    division == True
                    
            # Get the frames for cell 1 and frames/label for cell 2
            frames_1 = self._fetch_frames(j, division=division)
            
            # For frame_2, choose the next frame cell 1 appears in 
            last_frame_1 = np.amax(frames_1)
            frame_2 = np.amin( [x for x in track_id['frames'] if x > last_frame_1] )
            frames_2 = [frame_2]

            different_cells = track_id['different'][frame_2]
                    
            if type_cell == 0:
                # If there are no different cells in the subsequent frame, we must choose 
                # the same cell
                if len(different_cells) == 0:
                    type_cell = 1
                else:
                    label_2 = np.random.choice(different_cells)

            if type_cell == 1:
                # If there is only 1 cell in frame_2, we can only choose the class to be same
                label_2 = label_1
                        
            if type_cell == 2:
                # There should always be 2 daughters but not always a valid label
                label_2 = int(daughters[np.random.random_integers(0, len(daughters)-1)])

            track_1 = j
            track_2 = self.reverse_track_ids[batch][label_2]

            # Get appearances and centroid data
            appearance_1 = self._fetch_appearances(track_1, frames_1)
            appearance_2 = self._fetch_appearances(track_2, frames_2)
            centroid_1 = self._fetch_centroids(track_1, frames_1)
            centroid_2 = self._fetch_centroids(track_2, frames_2)

            # Apply random transforms
            new_appearance_1 = np.zeros(appearance_1.shape, dtype = K.floatx())
            new_appearance_2 = np.zeros(appearance_2.shape, dtype = K.floatx())

            for frame in range(appearance_1.shape[self.time_axis-1]):
                if self.data_format == 'channels_first':
                    app_temp = self.image_data_generator.random_transform(appearance_1[:,frame,:,:])
                    app_temp = self.image_data_generator.standardize(app_temp)
                    new_appearance_1[:,frame,:,:] = app_temp                        

                if self.data_format == 'channels_last':
                    app_temp = self.image_data_generator.random_transform(appearance_1[frame])
                    app_temp = self.image_data_generator.standardize(app_temp)
                    new_appearance_1[frame] = app_temp

            if self.data_format == 'channels_first':
                app_temp = self.image_data_generator.random_transform(appearance_2[:,0,:,:])
                app_temp = self.image_data_generator.standardize(app_temp)
                new_appearance_2[:,0,:,:] = app_temp   
                
            if self.data_format == 'channels_last':
                app_temp = self.image_data_generator.random_transform(appearance_2[0])
                app_temp = self.image_data_generator.standardize(app_temp)
                new_appearance_2[0] = app_temp

            # Compute distances between centroids
            centroids = np.concatenate([centroid_1, centroid_2], axis=0)
            distance = np.diff(centroids, axis=0)
            zero_pad = np.zeros((1,2), dtype = K.floatx())
            distance = np.concatenate([zero_pad, distance], axis=0)

            # Save images and distances to the batch arrays
            batch_x_1[i] = appearance_1
            batch_x_2[i] = appearance_2
            batch_distance_1[i] = distance[0:-1,:]
            batch_distance_2[i,0,:] = distance[-1,:]
            batch_y[i, type_cell] = 1
   
        # Remove singleton dimensions (if min_track_length is 1)
        if self.squeeze:
            batch_x_1 = np.squeeze(batch_x_1, axis=self.time_axis)
            batch_x_2 = np.squeeze(batch_x_2, axis=self.time_axis)
            batch_distance_1 = np.squeeze(batch_distance_1, axis=1)
            batch_distance_2 = np.squeeze(batch_distance_2, axis=1)
        return [batch_x_1, batch_x_2, batch_distance_1, batch_distance_2], batch_y

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


"""
Watershed generator
"""


class WatershedDataGenerator(ImageFullyConvDataGenerator):
    def flow(self,
             train_dict,
             batch_size=1,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             distance_bins=16,
             save_format='png'):
        return WatershedIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            distance_bins=distance_bins,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class WatershedIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 distance_bins=16,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `WatershedIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.distance_bins = distance_bins
        self.y = train_dict['y']
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(WatershedIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        if self.channel_axis == 1:
            batch_y = np.zeros(tuple([len(index_array), self.distance_bins] +
                                     list(self.y.shape)[2:]))
        else:
            batch_y = np.zeros(tuple([len(index_array)] +
                                     list(self.y.shape)[1:3] +
                                     [self.distance_bins]))

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            # TODO: hardcoded for cell interior = 1
            if self.channel_axis == 1:
                interior = y[1, :, :]
            else:
                interior = y[:, :, 1]

            distance = distance_transform_2d(interior, self.distance_bins)

            # convert to one hot notation
            if self.channel_axis == 1:
                y_shape = (self.distance_bins, self.y.shape[2], self.y.shape[3])
            else:
                y_shape = (self.y.shape[1], self.y.shape[2], self.distance_bins)

            y = np.zeros(y_shape)
            for label_val in range(np.amax(distance) + 1):
                if self.channel_axis == 1:
                    y[label_val, :, :] = distance == label_val
                else:
                    y[:, :, label_val] = distance == label_val

            batch_x[i] = x
            batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                # Save X batch
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                if self.y is not None:
                    # Save y batch, but just the MAX distance for each pixel
                    img_y = np.argmax(batch_y[i], axis=self.channel_axis - 1)
                    img_y = np.expand_dims(img_y, axis=self.channel_axis - 1)
                    img = array_to_img(img_y, self.data_format, scale=True)
                    fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

"""
Custom movie generators
"""


class MovieDataGenerator(ImageDataGenerator):
    """Generate minibatches of movie data with real-time data augmentation."""

    def __init__(self, **kwargs):
        super(MovieDataGenerator, self).__init__(**kwargs)
        # Change the axes for 5D data
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 3
            self.col_axis = 4
            self.time_axis = 2
        if self.data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3
            self.time_axis = 1

    def flow(self,
             train_dict,
             batch_size=1,
             frames_per_batch=10,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return MovieArrayIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            frames_per_batch=frames_per_batch,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        # TODO: standardize each image, not all frames at once
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')

        return x

    def random_transform(self, x, labels=None, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 5D tensor, image stack.
            labels: 5D tensor, image mask stack.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_time_axis = self.time_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.deg2rad(
                np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(
                transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([
                [1, -np.sin(shear), 0],
                [0, np.cos(shear), 0],
                [0, 0, 1]
            ])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(
                transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([
                [zx, 0, 0],
                [0, zy, 0],
                [0, 0, 1]
            ])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(
                transform_matrix, zoom_matrix)

        if labels is not None:
            y = labels

            if transform_matrix is not None:
                y_new = []
                h, w = y.shape[img_row_axis], y.shape[img_col_axis]
                transform_matrix_y = transform_matrix_offset_center(transform_matrix, h, w)
                for frame in range(y.shape[img_time_axis]):
                    if self.time_axis == 2:
                        y_frame = y[:, frame]
                        trans_channel_axis = img_channel_axis
                    else:
                        y_frame = y[frame]
                        trans_channel_axis = img_channel_axis - 1
                    y_trans = apply_transform(y_frame, transform_matrix_y, trans_channel_axis,
                                              fill_mode='constant', cval=0)
                    y_new.append(np.rint(y_trans))
                y = np.stack(y_new, axis=img_time_axis)

        if transform_matrix is not None:
            x_new = []
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix_x = transform_matrix_offset_center(transform_matrix, h, w)
            for frame in range(x.shape[img_time_axis]):
                if self.time_axis == 2:
                    x_frame = x[:, frame]
                    trans_channel_axis = img_channel_axis
                else:
                    x_frame = x[frame]
                    trans_channel_axis = img_channel_axis - 1
                x_trans = apply_transform(x_frame, transform_matrix_x, trans_channel_axis,
                                          fill_mode=self.fill_mode, cval=self.cval)
                x_new.append(x_trans)
            x = np.stack(x_new, axis=img_time_axis)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                if labels is not None:
                    y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                if labels is not None:
                    y = flip_axis(y, img_row_axis)

        if labels is not None:
            return x, y

        return x

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 5.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 4.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.time_axis, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.time_axis, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())


class MovieArrayIterator(Iterator):
    """
    The movie array iterator takes in a dictionary containing the training data
    Each data set contains a data movie (X) and a label movie (y)
    The label movie is the same dimension as the channel movie with each pixel
    having its corresponding prediction
    """

    def __init__(self,
                 train_dict,
                 movie_data_generator,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 frames_per_batch=10,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):

        if train_dict['y'] is not None and train_dict['X'].shape[0] != train_dict['y'].shape[0]:
            raise ValueError('`X` (movie data) and `y` (labels) '
                             'should have the same size. Found '
                             'Found x.shape = {}, y.shape = {}'.format(
                                 train_dict['X'].shape, train_dict['y'].shape))

        if data_format is None:
            data_format = K.image_data_format()

        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())
        self.y = np.asarray(train_dict['y'], dtype=np.int32)

        if self.x.ndim != 5:
            raise ValueError('Input data in `MovieArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)

        if self.x.shape[self.time_axis] - frames_per_batch < 0:
            raise ValueError(
                'The number of frames used in each training batch should '
                'be less than the number of frames in the training data!')

        self.frames_per_batch = frames_per_batch
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(MovieArrayIterator, self).__init__(
            len(self.y), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format == 'channels_first':
            batch_x = np.zeros((len(index_array),
                                self.x.shape[1],
                                self.frames_per_batch,
                                self.x.shape[3],
                                self.x.shape[4]))
            if self.y is not None:
                batch_y = np.zeros((len(index_array),
                                    self.y.shape[1],
                                    self.frames_per_batch,
                                    self.y.shape[3],
                                    self.y.shape[4]))

        else:
            batch_x = np.zeros(tuple([len(index_array), self.frames_per_batch] +
                                     list(self.x.shape)[2:]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array), self.frames_per_batch] +
                                         list(self.y.shape)[2:]))

        for i, j in enumerate(index_array):
            if self.y is not None:
                y = self.y[j]

            # Sample along the time axis
            last_frame = self.x.shape[self.time_axis] - self.frames_per_batch
            time_start = np.random.randint(0, high=last_frame)
            time_end = time_start + self.frames_per_batch
            if self.time_axis == 1:
                x = self.x[j, time_start:time_end, :, :, :]
                if self.y is not None:
                    y = self.y[j, time_start:time_end, :, :, :]

            elif self.time_axis == 2:
                x = self.x[j, :, time_start:time_end, :, :]
                if self.y is not None:
                    y = self.y[j, :, time_start:time_end, :, :]

            if self.y is not None:
                x, y = self.movie_data_generator.random_transform(
                    x.astype(K.floatx()), labels=y)
                x = self.movie_data_generator.standardize(x)
                batch_y[i] = y
            else:
                x = self.movie_data_generator.random_transform(x.astype(K.floatx()))

            batch_x[i] = x

        if self.save_to_dir:
            time_axis = 2 if self.data_format == 'channels_first' else 1
            for i, j in enumerate(index_array):
                for frame in range(batch_x.shape[time_axis]):
                    if time_axis == 2:
                        img = array_to_img(batch_x[i, :, frame], self.data_format, scale=True)
                    else:
                        img = array_to_img(batch_x[i, frame], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

                    if self.y is not None:
                        if time_axis == 2:
                            img_y = array_to_img(batch_y[i, :, frame], self.data_format, scale=True)
                        else:
                            img_y = array_to_img(batch_y[i, frame], self.data_format, scale=True)
                        fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e4),
                            format=self.save_format)
                        img_y.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class SampleMovieArrayIterator(Iterator):
    def __init__(self,
                 train_dict,
                 movie_data_generator,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if train_dict['y'] is not None and train_dict['X'].shape[0] != train_dict['y'].shape[0]:
            raise ValueError('`X` (movie data) and `y` (labels) '
                             'should have the same size. Found '
                             'Found x.shape = {}, y.shape = {}'.format(
                                 train_dict['X'].shape, train_dict['y'].shape))

        if data_format is None:
            data_format = K.image_data_format()

        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())
        self.y = np.asarray(train_dict['y'], dtype=np.int32)

        if self.x.ndim != 5:
            raise ValueError('Input data in `SampleMovieArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)

        self.win_x = train_dict['win_x']
        self.win_y = train_dict['win_y']
        self.win_z = train_dict['win_z']
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(SampleMovieArrayIterator, self).__init__(
            len(self.y), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.channel_axis == 1:
            batch_x = np.zeros((len(index_array),
                                self.x.shape[self.channel_axis],
                                2 * self.win_z + 1,
                                2 * self.win_x + 1,
                                2 * self.win_y + 1))
        else:
            batch_x = np.zeros((len(index_array),
                                2 * self.win_z + 1,
                                2 * self.win_x + 1,
                                2 * self.win_y + 1,
                                self.x.shape[self.channel_axis]))

        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.movie_data_generator.random_transform(x.astype(K.floatx()))
            x = self.movie_data_generator.standardize(x)

            batch_x[i] = x

        if self.save_to_dir:
            time_axis = 2 if self.data_format == 'channels_first' else 1
            for i, j in enumerate(index_array):
                for frame in range(batch_x.shape[time_axis]):
                    if time_axis == 2:
                        img = batch_x[i, :, frame]
                    else:
                        img = batch_x[i, frame]
                    img = array_to_img(img, self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class SampleMovieDataGenerator(MovieDataGenerator):
    def flow(self,
             train_dict,
             batch_size=32,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return SampleMovieArrayIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class WatershedMovieDataGenerator(MovieDataGenerator):
    """
    Generate minibatches of distance-transformed movie data
    with real-time data augmentation.
    """
    def flow(self,
             train_dict,
             batch_size=1,
             shuffle=True,
             seed=None,
             frames_per_batch=10,
             save_to_dir=None,
             save_prefix='',
             distance_bins=16,
             erosion_width=None,
             save_format='png'):
        return WatershedMovieIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            frames_per_batch=frames_per_batch,
            data_format=self.data_format,
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class WatershedMovieIterator(Iterator):
    def __init__(self,
                 train_dict,
                 movie_data_generator,
                 batch_size=1,
                 shuffle=False,
                 seed=None,
                 frames_per_batch=10,
                 data_format=None,
                 distance_bins=16,
                 erosion_width=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()

        if train_dict['y'] is not None and train_dict['X'].shape[0] != train_dict['y'].shape[0]:
            raise ValueError('`X` (movie data) and `y` (labels) '
                             'should have the same size. Found '
                             'Found x.shape = {}, y.shape = {}'.format(
                                 train_dict['X'].shape, train_dict['y'].shape))

        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 5:
            raise ValueError('Input data in `WatershedMovieIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)

        if self.x.shape[self.time_axis] < frames_per_batch:
            raise ValueError('The number of frames used in each training batch '
                             'should be less than the number of frames in the '
                             'training data.')

        self.frames_per_batch = frames_per_batch
        self.distance_bins = distance_bins
        self.erosion_width = erosion_width
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        if self.data_format == 'channels_first':
            y_shape = (train_dict['y'].shape[0], *train_dict['y'].shape[2:])
        else:
            y_shape = train_dict['y'].shape[0:-1]

        y_distance = np.zeros(y_shape)
        for batch in range(y_distance.shape[0]):
            # hardcoded for image mask
            if self.time_axis == 1:
                mask = train_dict['y'][batch, :, :, :, 0]
            else:
                mask = train_dict['y'][batch, 0, :, :, :]
            y_distance[batch] = distance_transform_3d(mask, self.distance_bins, self.erosion_width)
        # convert to one hot notation
        y_distance = keras_to_categorical(np.expand_dims(y_distance, axis=-1))
        if self.channel_axis == 1:
            y_distance = np.rollaxis(y_distance, -1, 1)
        self.y = y_distance
        super(WatershedMovieIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format == 'channels_first':
            batch_x = np.zeros((len(index_array),
                                self.x.shape[1],
                                self.frames_per_batch,
                                self.x.shape[3],
                                self.x.shape[4]))
            if self.y is not None:
                batch_y = np.zeros((len(index_array),
                                    self.distance_bins,
                                    self.frames_per_batch,
                                    self.y.shape[3],
                                    self.y.shape[4]))

        else:
            batch_x = np.zeros(tuple([len(index_array), self.frames_per_batch] +
                                     list(self.x.shape)[2:]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array), self.frames_per_batch] +
                                         list(self.y.shape)[2:4] +
                                         [self.distance_bins]))

        for i, j in enumerate(index_array):
            # Sample along the time axis
            last_frame = self.x.shape[self.time_axis] - self.frames_per_batch
            time_start = np.random.randint(0, high=last_frame)
            time_end = time_start + self.frames_per_batch

            if self.time_axis == 1:
                x = self.x[j, time_start:time_end, :, :, :]
                if self.y is not None:
                    y = self.y[j, time_start:time_end, :, :, :]

            elif self.time_axis == 2:
                x = self.x[j, :, time_start:time_end, :, :]
                if self.y is not None:
                    y = self.y[j, :, time_start:time_end, :, :]

            if self.y is not None:
                x, y = self.movie_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.movie_data_generator.random_transform(x.astype(K.floatx()))

            x = self.movie_data_generator.standardize(x)

            batch_x[i] = x
            if self.y is not None:
                batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                for frame in range(batch_x.shape[self.time_axis]):
                    if self.time_axis == 2:
                        img = batch_x[i, :, frame]
                    else:
                        img = batch_x[i, frame]
                    img = array_to_img(img, self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

                    if self.y is not None:
                        # Save y batch, but just the MAX distance for each pixel
                        if self.time_axis == 2:
                            img_y = np.argmax(batch_y[i, :, frame], axis=0)
                            img_channel_axis = 0
                            img_y = batch_y[i, :, frame]
                        else:
                            img_channel_axis = -1
                            img_y = batch_y[i, frame]
                        img_y = np.argmax(img_y, axis=img_channel_axis)
                        img_y = np.expand_dims(img_y, axis=img_channel_axis)
                        img = array_to_img(img_y, self.data_format, scale=True)
                        fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e4),
                            format=self.save_format)
                        img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


"""
Bounding box generators adapted from retina net library
"""


class BoundingBoxIterator(Iterator):
    def __init__(self, train_dict, image_data_generator,
                 batch_size=1, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `BoundingBoxIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.y = train_dict['y']

        if self.channel_axis == 3:
            self.num_features = self.y.shape[-1]
        else:
            self.num_features = self.y.shape[1]

        if self.channel_axis == 3:
            self.image_shape = self.x.shape[1:2]
        else:
            self.image_shape = self.x.shape[2:]

        bbox_list = []
        for b in range(self.x.shape[0]):
            for l in range(1, self.num_features - 1):
                if self.channel_axis == 3:
                    mask = self.y[b, :, :, l]
                else:
                    mask = self.y[b, l, :, :]
                props = regionprops(label(mask))
                bboxes = [np.array(list(prop.bbox) + list(l)) for prop in props]
                bboxes = np.concatenate(bboxes, axis=0)
            bbox_list.append(bboxes)
        self.bbox_list = bbox_list

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(BoundingBoxIterator, self).__init__(self.x.shape[0], batch_size, shuffle, seed)

    def get_annotations(self, y):
        for l in range(1, self.num_features - 1):
            if self.channel_axis == 3:
                mask = y[:, :, l]
            else:
                mask = y[l, :, :]
            props = regionprops(label(mask))
            bboxes = [np.array(list(prop.bbox) + list(l)) for prop in props]
            bboxes = np.concatenate(bboxes, axis=0)
        return bboxes

    def anchor_targets(self,
                       image_shape,
                       annotations,
                       num_classes,
                       mask_shape=None,
                       negative_overlap=0.4,
                       positive_overlap=0.5,
                       **kwargs):
        return self.anchor_targets_bbox(
            image_shape,
            annotations,
            num_classes,
            mask_shape,
            negative_overlap,
            positive_overlap,
            **kwargs)

    def compute_target(self, annotation):
        labels, annotations, anchors = self.anchor_targets(
            self.image_shape, annotation, self.num_features)
        regression = self.bbox_transform(anchors, annotation)

        # append anchor state to regression targets
        anchor_states = np.max(labels, axis=1, keepdims=True)
        regression = np.append(regression, anchor_states, axis=1)
        return [regression, labels]

    def _get_batches_of_transformed_samples(self, index_array):
        index_array = index_array[0]
        if self.channel_axis == 1:
            batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:4]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape)[1:4]))
        else:
            batch_x = np.zeros((len(index_array),
                                self.x.shape[2],
                                self.x.shape[3],
                                self.x.shape[1]))
            if self.y is not None:
                batch_y = np.zeros((len(index_array),
                                    self.y.shape[2],
                                    self.y.shape[3],
                                    self.y.shape[1]))

        regressions_list = []
        labels_list = []

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            if self.channel_axis == 1:
                batch_x[i] = x
                batch_y[i] = y

                # Get the bounding boxes from the transformed masks!
                annotations = self.get_annotations(y)
                regressions, labels = self.compute_target(annotations)
                regressions_list.append(regressions)
                labels_list.append(labels)

            if self.channel_axis == 3:
                raise NotImplementedError('Bounding box generator does not work '
                                          'for channels last yet')

            regressions = np.stack(regressions_list, axis=0)
            labels = np.stack(labels_list, axis=0)

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        return batch_x, [regressions_list, labels_list]

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


"""
Discriminative generator
"""


class DiscDataGenerator(ImageFullyConvDataGenerator):

    def flow(self,
             train_dict,
             batch_size=1,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return DiscIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class DiscIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `DiscIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.y = train_dict['y']
        self.max_label = 0
        for batch in range(self.y.shape[0]):
            if self.channel_axis == 1:
                label_matrix = label(self.y[batch, 1, :, :])
            else:
                label_matrix = label(self.y[batch, :, :, 1])
            max_label = np.amax(label_matrix)
            if max_label > self.max_label:
                self.max_label = max_label

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(DiscIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        if self.channel_axis == 1:
            batch_y = np.zeros(tuple([len(index_array), self.max_label + 1] +
                                     list(self.y.shape)[2:]))
        else:
            batch_y = np.zeros(tuple([len(index_array)] +
                                     list(self.y.shape)[1:3] +
                                     [self.max_label + 1]))

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            # TODO: hardcoded for cell interior = 1
            if self.channel_axis == 1:
                interior = y[1, :, :]
            else:
                interior = y[:, :, 1]

            label_matrix = label(interior)

            # convert to one hot notation
            if self.channel_axis == 1:
                y_shape = (self.max_label + 1, self.y.shape[2], self.y.shape[3])
            else:
                y_shape = (self.y.shape[1], self.y.shape[2], self.max_label + 1)

            y = np.zeros(y_shape)

            for label_val in range(self.max_label + 1):
                if self.channel_axis == 1:
                    y[label_val, :, :] = label_matrix == label_val
                else:
                    y[:, :, label_val] = label_matrix == label_val

            batch_x[i] = x
            batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                # Save X batch
                img_x = np.expand_dims(batch_x[i, :, :, 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                if self.y is not None:
                    img_y = np.expand_dims(batch_y[i, :, :, 0], -1)
                    img = array_to_img(img_y, self.data_format, scale=True)
                    fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


"""
RetinaNet and MaskRCNN Generators
"""


class RetinaNetGenerator(_RetinaNetGenerator):

    def __init__(self,
                 direc_name,
                 training_dirs,
                 raw_image_dir,
                 channel_names,
                 annotation_dir,
                 annotation_names,
                 **kwargs):
        self.image_names = []
        self.image_data = {}
        self.image_stack = []
        self.mask_stack = []
        self.base_dir = kwargs.get('base_dir')

        train_files = self.list_file_deepcell(
            dir_name=direc_name,
            training_dirs=training_dirs,
            image_dir=raw_image_dir,
            channel_names=channel_names)

        annotation_files = self.list_file_deepcell(
            dir_name=direc_name,
            training_dirs=training_dirs,
            image_dir=annotation_dir,
            channel_names=annotation_names)

        self.image_stack = self.generate_subimage(train_files, 3, 3, True)
        self.mask_stack = self.generate_subimage(annotation_files, 3, 3, False)

        self.classes = {'cell': 0}

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_data = self._read_annotations(self.mask_stack)
        self.image_names = list(self.image_data.keys())
        super(RetinaNetGenerator, self).__init__(**kwargs)

    def list_file_deepcell(self, dir_name, training_dirs, image_dir, channel_names):
        """
        List all image files inside each `dir_name/training_dir/image_dir`
        with "channel_name" in the filename.
        """
        filelist = []
        for direc in training_dirs:
            imglist = os.listdir(os.path.join(dir_name, direc, image_dir))

            for channel in channel_names:
                for img in imglist:
                    # if channel string is NOT in image file name, skip it.
                    if not fnmatch(img, '*{}*'.format(channel)):
                        continue
                    image_file = os.path.join(dir_name, direc, image_dir, img)
                    filelist.append(image_file)
        return sorted(filelist)

    def _read_annotations(self, masks_list):
        result = {}
        for cnt, image in enumerate(masks_list):
            result[cnt] = []
            p = regionprops(label(image))

            cell_count = 0
            for index in range(len(np.unique(label(image))) - 1):
                y1, x1, y2, x2 = p[index].bbox
                result[cnt].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
                cell_count += 1
            if cell_count == 0:
                logging.warning('No cells found in image {}'.format(cnt))
        return result

    def generate_subimage(self, img_pathstack, horizontal, vertical, flag):
        sub_img = []
        for img_path in img_pathstack:
            img = np.asarray(np.float32(imread(img_path)))
            if flag:
                img = (img / np.max(img))
            vway = np.zeros(vertical + 1)  # The dimentions of vertical cuts
            hway = np.zeros(horizontal + 1)  # The dimentions of horizontal cuts
            vcnt = 0  # The initial value for vertical
            hcnt = 0  # The initial value for horizontal

            for i in range(vertical + 1):
                vway[i] = int(vcnt)
                vcnt += (img.shape[1] / vertical)

            for j in range(horizontal + 1):
                hway[j] = int(hcnt)
                hcnt += (img.shape[0] / horizontal)

            vb = 0

            for i in range(len(hway) - 1):
                for j in range(len(vway) - 1):
                    vb += 1

            for i in range(len(hway) - 1):
                for j in range(len(vway) - 1):
                    s = img[int(hway[i]):int(hway[i + 1]), int(vway[j]):int(vway[j + 1])]
                    sub_img.append(s)

        if flag:
            sub_img = [np.tile(np.expand_dims(i, axis=-1), (1, 1, 3)) for i in sub_img]

        return sub_img

    def size(self):
        """Size of the dataset."""
        return len(self.image_names)

    def num_classes(self):
        """Number of classes in the dataset."""
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """Map name to label."""
        return self.classes[name]

    def label_to_name(self, label):
        """Map label to name."""
        return self.labels[label]

    def image_path(self, image_index):
        """Returns the image path for image_index."""
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """Compute the aspect ratio for an image with image_index."""
        image = self.image_stack[image_index]
        return float(image.shape[1]) / float(image.shape[0])

    def load_image(self, image_index):
        """Load an image at the image_index."""
        return self.image_stack[image_index]

    def load_annotations(self, image_index):
        """Load annotations for an image_index."""
        path = self.image_names[image_index]
        annots = self.image_data[path]
        boxes = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = 'cell'
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes


class MaskRCNNGenerator(_MaskRCNNGenerator):
    def __init__(self,
                 direc_name,
                 training_dirs,
                 raw_image_dir,
                 channel_names,
                 annotation_dir,
                 annotation_names,
                 base_dir=None,
                 image_min_side=200,
                 image_max_side=200,
                 crop_iterations=1,
                 **kwargs):
        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir
        self.image_stack = []

        train_files = self.list_file_deepcell(
            dir_name=direc_name,
            training_dirs=training_dirs,
            image_dir=raw_image_dir,
            channel_names=channel_names)

        annotation_files = self.list_file_deepcell(
            dir_name=direc_name,
            training_dirs=training_dirs,
            image_dir=annotation_dir,
            channel_names=annotation_names)

        store = self.randomcrops(
            train_files,
            annotation_files,
            image_min_side,
            image_max_side,
            iteration=crop_iterations)

        self.image_stack = store[0]
        self.classes = {'cell': 0}

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_data = self._read_annotations(store[1])

        self.image_names = list(self.image_data.keys())

        # Override default Generator value with custom anchor_targets_bbox
        if 'compute_anchor_targets' not in kwargs:
            kwargs['compute_anchor_targets'] = anchor_targets_bbox

        super(MaskRCNNGenerator, self).__init__(
            image_min_side=image_min_side,
            image_max_side=image_max_side,
            **kwargs)

    def list_file_deepcell(self, dir_name, training_dirs, image_dir, channel_names):
        """
        List all image files inside each `dir_name/training_dir/image_dir`
        with "channel_name" in the filename.
        """
        filelist = []
        for direc in training_dirs:
            imglist = os.listdir(os.path.join(dir_name, direc, image_dir))

            for channel in channel_names:
                for img in imglist:
                    # if channel string is NOT in image file name, skip it.
                    if not fnmatch(img, '*{}*'.format(channel)):
                        continue
                    image_file = os.path.join(dir_name, direc, image_dir, img)
                    filelist.append(image_file)
        return sorted(filelist)

    def randomcrops(self, dirpaths, maskpaths, size_x, size_y, iteration=1):
        img = cv2.imread(dirpaths[0], 0)
        img_y = img.shape[0]
        img_x = img.shape[1]
        act_x = img_x - size_x
        act_y = img_y - size_y
        if act_x < 0 or act_y < 0:
            logging.warning('Image to crop is of a smaller size')
            return ([], [])
        outputi = []
        outputm = []
        while iteration > 0:
            cropindex = []
            for path in dirpaths:
                rand_x = np.random.randint(0, act_x)
                rand_y = np.random.randint(0, act_y)
                cropindex.append((rand_x, rand_y))
                image = cv2.imread(path, 0)
                newimg = image[rand_y:rand_y + size_y, rand_x:rand_x + size_x]
                newimg = np.tile(np.expand_dims(newimg, axis=-1), (1, 1, 3))
                outputi.append(newimg)

            for i, path in enumerate(maskpaths):
                image = cv2.imread(path, 0)
                rand_x = cropindex[i][0]
                rand_y = cropindex[i][1]
                newimg = image[rand_y:rand_y + size_y, rand_x:rand_x + size_x]
                outputm.append(newimg)

            iteration -= 1
        return (outputi, outputm)

    def _read_annotations(self, maskarr):
        result = {}
        for cnt, image in enumerate(maskarr):
            result[cnt] = []
            l = label(image)
            p = regionprops(l)
            cell_count = 0
            for index in range(len(np.unique(l)) - 1):
                y1, x1, y2, x2 = p[index].bbox
                result[cnt].append({
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                    'class': 'cell',
                    'mask_path': np.where(l == index + 1, 1, 0)
                })
                cell_count += 1
            print('Image number {} has {} cells'.format(cnt, cell_count))
            # If there are no cells in this image, remove it from the annotations
            if not result[cnt]:
                del result[cnt]
        return result

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        # image = Image.open(self.image_path(image_index))
        # return float(image.width) / float(image.height)
        image = self.image_stack[image_index]
        return float(image.shape[1]) / float(image.shape[0])

    def load_image(self, image_index):
        # return read_image_bgr(self.image_path(image_index))
        return self.image_stack[image_index]

    def load_annotations(self, image_index):
        path = self.image_names[image_index]
        annots = self.image_data[path]

        # find mask size in order to allocate the right dimension for the annotations
        annotations = np.zeros((len(annots), 5))
        masks = []

        for idx, annot in enumerate(annots):
            annotations[idx, 0] = float(annot['x1'])
            annotations[idx, 1] = float(annot['y1'])
            annotations[idx, 2] = float(annot['x2'])
            annotations[idx, 3] = float(annot['y2'])
            annotations[idx, 4] = self.name_to_label(annot['class'])
            mask = annot['mask_path']
            mask = (mask > 0).astype(np.uint8)  # convert from 0-255 to binary mask
            masks.append(np.expand_dims(mask, axis=-1))

        return annotations, masks
