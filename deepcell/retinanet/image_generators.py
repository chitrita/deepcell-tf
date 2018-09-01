"""
image_generators.py
Image generators for training retinanets
@author: David Van Valen
"""

import numpy as np
import random
import threading
import warnings
import keras

from keras_retinanet.utils.anchors import (
    anchor_targets_bbox,
    bbox_transform,
    anchors_for_shape,
    guess_shapes
)
from keras_retinanet.utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from keras_retinanet.utils.transform import transform_aabb


class Generator(object):
    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        compute_shapes=guess_shapes,
        compute_anchor_targets=anchor_targets_bbox,
    ):
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_shapes         = compute_shapes
        self.compute_anchor_targets = compute_anchor_targets

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, masks_group, group):
        # test all annotations
        for index, (image, annotations, masks) in enumerate(zip(image_group, annotations_group, masks_group)):
            assert(isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # check if all masks have the same size of the respective image
            for idx in range(len(masks)):
                assert(image.shape[:2] == masks[idx].shape[:2]), 'Found different image ({}) and mask ({}) size in image {}'.format(image.shape, masks[idx].shape, group[index])

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0]) |
                (np.sum(masks, axis=(1,2,3)) == 0)
            )[0]

            # delete invalid indices
            if len(invalid_indices):
#                 warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
#                     group[index],
#                     image.shape,
#                     [annotations[invalid_index, :] for invalid_index in invalid_indices]
#                 ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)
                masks_group[index]       = np.delete(masks, invalid_indices, axis=0)

        return image_group, annotations_group, masks_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, masks):
        # randomly transform both image and annotations
        if self.transform_generator:
            transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)
            image     = apply_transform(transform, image, self.transform_parameters)

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
                
            # randomly transform the masks and expand so to have a fake channel dimension
            for m in range(len(masks)):
                masks[m] = apply_transform(transform, masks[m], self.transform_parameters)
                masks[m] = np.expand_dims(masks[m], axis=-1)

            # randomly transform the bounding boxes
            annotations = annotations.copy()
            for index in range(annotations.shape[0]):
                annotations[index, :4] = transform_aabb(transform, annotations[index, :4])

        return image, annotations, masks

    def resize_image(self, image):
        resized_image, scale = resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)
        if len(resized_image.shape) == 2:
            resized_image = np.expand_dims(resized_image, axis=-1)
        return resized_image, scale
        
    def preprocess_image(self, image):
        return image

    def preprocess_group_entry(self, image, annotations, masks):
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations, masks = self.random_transform_group_entry(image, annotations, masks)

        # resize image
        image, image_scale = self.resize_image(image)

        # resize masks
        for i in range(len(masks)):
            masks[i], _ = self.resize_image(masks[i])

        # apply resizing to annotations too
        annotations[:, :4]  *= image_scale

        return image, annotations, masks

    def preprocess_group(self, image_group, annotations_group, masks_group):
        for index, (image, annotations, masks) in enumerate(zip(image_group, annotations_group, masks_group)):
            # preprocess a single group entry
            image, annotations, masks = self.preprocess_group_entry(image, annotations, masks)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations
            masks_group[index]       = masks

        return image_group, annotations_group, masks_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def generate_anchors(self, image_shape):
        return anchors_for_shape(image_shape, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group, masks_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        labels_batch, regression_batch, _ = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        # copy all annotations / masks to the batch
        max_annotations = max(a.shape[0] for a in annotations_group)
        # masks_batch has shape: (batch size, max_annotations, bbox_x1 + bbox_y1 + bbox_x2 + bbox_y2 + prediction_label + width + height + max_image_dimension)
        masks_batch     = np.zeros((self.batch_size, max_annotations, 5 + 2 + max_shape[0] * max_shape[1]), dtype=keras.backend.floatx())
        for index, (annotations, masks) in enumerate(zip(annotations_group, masks_group)):
            masks_batch[index, :annotations.shape[0], :annotations.shape[1]] = annotations
            masks_batch[index, :, 5] = max_shape[1]  # width
            masks_batch[index, :, 6] = max_shape[0]  # height

            # add flattened mask
            for mask_index, mask in enumerate(masks):
                masks_batch[index, mask_index, 7:] = mask.flatten()

        return [regression_batch, labels_batch, masks_batch]

    def compute_input_output(self, group):
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # split annotations and masks
        masks_group       = [m for _, m in annotations_group]
        annotations_group = [a for a, _ in annotations_group]

        # perform preprocessing steps
        image_group, annotations_group, masks_group = self.preprocess_group(image_group, annotations_group, masks_group)

        # check validity of annotations again - augmentations can cause cells to disappear
        image_group, annotations_group, masks_group = self.filter_annotations(image_group, annotations_group, masks_group, group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group, masks_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)

class MaskRCNNGenerator(Generator):
    def __init__(self,
                 train_dict,
                 image_min_side=200,
                 image_max_side=200,
                 data_format=None,
                 **kwargs):

        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        
        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.row_axis = 1 if data_format == 'channels_last' else 2
        self.col_axis = 2 if data_format == 'channels_last' else 3
        
        self.x = train_dict['X']
        self.y = train_dict['y']
        
        if len(self.x.shape) == 5:
            x,y = self.x, self.y
            if self.data_format == 'channels_last':
                new_shape = (x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
            else:
                new_shape = (x.shape[0]*x.shape[2], x.shape[1], x.shape[3], x.shape[4])
            x_temp = np.reshape(x, new_shape)
            y_temp = np.reshape(y, new_shape)
            
            self.x = x_temp
            self.y = y_temp
                        
        # Make images square
        if self.x.shape[self.row_axis] != self.x.shape[self.col_axis]:
            axis_shape = np.amin([self.x.shape[self.row_axis], self.x.shape[self.col_axis]])
            
            if data_format == 'channels_last':
                self.x = self.x[:,0:axis_shape,0:axis_shape,:]
                self.y = self.y[:,0:axis_shape,0:axis_shape,:]
            else:
                self.x = self.x[:,:,0:axis_shape,0:axis_shape]
                self.y = self.y[:,:,0:axis_shape,0:axis_shape] 
                
        # Remove border cells
        for b in range(self.y.shape[0]):
            if self.data_format == 'channels_last':
                y_batch = self.y[b,:,:,0]
            else:
                y_batch = self.y[b,0,:,:]
            self.y[b] = np.expand_dims(clear_border(y_batch), axis=self.channel_axis)
        
        good_batch = []
        # Remove images with small numbers of cells
        for b in range(self.x.shape[0]):
            n_cells = len(np.unique(self.y[b]))-1
            if n_cells > 3:
                good_batch.append(b)
                
        good_batch = np.array(good_batch)
        self.x = self.x[good_batch,:,:,:]
        self.y = self.y[good_batch,:,:,:]
        
        self.classes = {'cell': 0}
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_data = self._read_annotations(self.y)
        self.image_names = list(self.image_data.keys())

        # Override default Generator value with custom anchor_targets_bbox
        if 'compute_anchor_targets' not in kwargs:
            kwargs['compute_anchor_targets'] = anchor_targets_bbox

        super(MaskRCNNGenerator, self).__init__(
            image_min_side=image_min_side,
            image_max_side=image_max_side,
            **kwargs)

    def _read_annotations(self, maskarr):
        result = {}
        for cnt, l in enumerate(maskarr):
            result[cnt] = []
            p = regionprops(l)
            cell_count = 0
            cell_labels = np.unique(l[l>0])
            for index in range(len(cell_labels)):
                y1, x1, y2, x2 = p[index].bbox
                cell_label = p[index].label
                result[cnt].append({
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                    'class': 'cell',
                    'mask_path': np.where(l == cell_label, 1, 0)
                })
                cell_count += 1
#             print('Image number {} has {} cells'.format(cnt, cell_count))
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
        image = self.x[image_index]
        return float(image.shape[1]) / float(image.shape[0])

    def load_image(self, image_index):
        return self.x[image_index]

    def load_annotations(self, image_index):
        annots = self.image_data[image_index]

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
            masks.append(mask)
            
        return annotations, masks
