import os
import h5py
from random import shuffle

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import params


class DataHandler:
  no_classes = {'nus_wide': 81,
              'ms_coco': 80}
  data_types = {'nus_wide': ('train', 'test'),
                'ms_coco': ('train', 'val')}

  preprocessor_aug = ImageDataGenerator(featurewise_center=True,
                                        horizontal_flip=True,
                                        width_shift_range=1.0/8,
                                        height_shift_range=1.0/8,
                                        zoom_range = 1.0/8,
                                        fill_mode='constant')
  preprocessor = ImageDataGenerator(featurewise_center=True)
  

  def __init__(self, dataset):
    self.dataset = dataset
    f_dataset = h5py.File(os.path.join('dataset', dataset + '.h5'), 'r')

    self.train_images = f_dataset[self.data_types[dataset][0] + '_images']
    self.train_image_shapes = f_dataset[self.data_types[dataset][0] + '_image_shapes']
    self.train_labels = f_dataset[self.data_types[dataset][0] + '_labels']

    self.val_images = f_dataset[self.data_types[dataset][1] + '_images']
    self.val_image_shapes = f_dataset[self.data_types[dataset][1] + '_image_shapes']
    self.val_labels = f_dataset[self.data_types[dataset][1] + '_labels']

    self.preprocessor_aug.mean = f_dataset['mean']
    self.preprocessor.mean = f_dataset['mean']


  def generator(self, mode, shuffle_batches=True):
    while True:
      if mode == 'train':
        no_examples = len(self.train_images)
      elif mode == 'val':
        no_examples = len(self.val_images)

      no_batches = int(np.ceil(float(no_examples) / params.batch_size))
      inds_batch = range(no_batches)
      if shuffle_batches:
        shuffle(inds_batch)

      for ind_batch in inds_batch:
        if ind_batch == no_batches - 1:
          inds = range(no_examples - params.batch_size, no_examples)
        else:
          inds = range(ind_batch * params.batch_size, (ind_batch + 1) * params.batch_size)
        
        if mode == 'train':
          images_flat = self.train_images[inds]
          image_shapes = self.train_image_shapes[inds]
          prep = self.preprocessor_aug
          labels_batch = self.train_labels[inds]
        elif mode == 'val':
          images_flat = self.val_images[inds]
          image_shapes = self.val_image_shapes[inds]
          prep = self.preprocessor
          labels_batch = self.val_labels[inds]

        images_batch = np.empty((params.batch_size, 224, 224, 3), dtype=np.float32)
        for ind_image, image_flat in enumerate(images_flat):
          shaped_image = image_flat.reshape(224, 224, 3)
          images_batch[ind_image] = prep.random_transform(shaped_image)

        yield images_batch, labels_batch