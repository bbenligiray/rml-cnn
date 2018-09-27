import os
import h5py
import copy
import random
random.seed(26)

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
  

  def __init__(self, dataset, labeled_ratio, corruption_ratio):
    self.dataset = dataset
    self.labeled_ratio = labeled_ratio
    self.corruption_ratio = corruption_ratio

    f_dataset = h5py.File(os.path.join('dataset', dataset + '.h5'), 'r')

    self.train_images = f_dataset[self.data_types[dataset][0] + '_images']
    self.train_image_shapes = f_dataset[self.data_types[dataset][0] + '_image_shapes']
    self.train_labels = f_dataset[self.data_types[dataset][0] + '_labels']

    self.val_images = f_dataset[self.data_types[dataset][1] + '_images']
    self.val_image_shapes = f_dataset[self.data_types[dataset][1] + '_image_shapes']
    self.val_labels = f_dataset[self.data_types[dataset][1] + '_labels']

    self.preprocessor_aug.mean = f_dataset['mean']
    self.preprocessor.mean = f_dataset['mean']

    # decide on which examples are labeled
    if self.labeled_ratio == 100:
      self.inds_labeled = range(len(self.train_labels))
      self.inds_unlabeled = []
    else:
      approx_labeled_ratio = self.labeled_ratio
      while True:   
        self.inds_labeled = []
        no_examples_per_class = np.sum(self.train_labels, axis=0)
        no_labeled_examples_per_class = np.round(no_examples_per_class * (approx_labeled_ratio / 100.0)).astype(np.int)
        
        for ind_class in range(self.train_labels.shape[1]):
          inds = np.where(self.train_labels[:, ind_class] == 1)[0].tolist()
          random.shuffle(inds)
          self.inds_labeled += inds[:no_labeled_examples_per_class[ind_class]]

        self.inds_labeled = list(set(self.inds_labeled))
        true_labeled_ratio = (float(len(self.inds_labeled)) / len(self.train_labels)) * 100

        # have 0.5% tolerance, err on the positive side
        if true_labeled_ratio > self.labeled_ratio + 5:
          approx_labeled_ratio -= 2.5
        elif true_labeled_ratio > self.labeled_ratio + 0.5:
          approx_labeled_ratio -= 0.25
        elif true_labeled_ratio < self.labeled_ratio:
          approx_labeled_ratio += 0.25
        else:
          break

      self.inds_unlabeled = [x for x in range(len(self.train_labels)) if x not in self.inds_labeled]

    # corrupt labels
    if self.corruption_ratio > 0:
      corrupted_labels = self.train_labels[:]
      inds_corrupted = copy.deepcopy(self.inds_labeled)
      random.shuffle(inds_corrupted)
      inds_corrupted = inds_corrupted[:int(len(inds_corrupted) * (self.corruption_ratio / 100.0))]

      for ind_labeled in inds_corrupted:
        # pick a random example to decide on the number of positives
        ind_sample = random.randint(0, len(self.train_labels) - 1)
        no_pos = np.sum(self.train_labels[ind_sample])

        corrupted_labels[ind_labeled] = 0
        for ind in range(no_pos):
          pos_class = random.randint(0, self.train_labels.shape[1] - 1)
          corrupted_labels[ind_labeled, pos_class] = 1
      
      self.train_labels = corrupted_labels     


  def generator(self, data_type, shuffle_batches=True):
    if data_type == 'train_mixed':
      while True:
        no_labeled = len(self.inds_labeled)
        no_unlabeled = len(self.inds_unlabeled)
        no_examples = no_labeled + no_unlabeled

        no_batches = int(np.ceil(float(no_examples) / params.batch_size))
        inds = range(no_batches * params.batch_size)
        inds[no_examples:] = range(len(inds) - no_examples)
        random.shuffle(inds)

        for ind_batch in range(no_batches):
          inds_image = inds[ind_batch * params.batch_size:(ind_batch + 1) * params.batch_size]

          images_flat = np.empty((params.batch_size, self.train_images[0].shape[0]), dtype=np.float32)
          image_shapes = np.empty((params.batch_size, 3), dtype=np.float32)
          prep = self.preprocessor_aug
          labels_batch = np.empty((params.batch_size, self.train_labels.shape[1] + 1), dtype=np.float32)

          for ind, ind_image in enumerate(inds_image):
            if ind_image < no_labeled:
              images_flat[ind] = self.train_images[self.inds_labeled][ind_image]
              image_shapes[ind] = self.train_image_shapes[self.inds_labeled][ind_image]
              labels_batch[ind] = self.train_labels[self.inds_labeled][ind_image]
            else:
              ind_image -= no_labeled
              images_flat[ind] = self.train_images[self.inds_unlabeled][ind_image]
              image_shapes[ind] = self.train_image_shapes[self.inds_unlabeled][ind_image]
              labels_batch[ind] = self.train_labels[self.inds_unlabeled][ind_image]

          images_batch = np.empty((params.batch_size, 224, 224, 3), dtype=np.float32)
          for ind_image, image_flat in enumerate(images_flat):
            shaped_image = image_flat.reshape(224, 224, 3)
            images_batch[ind_image] = prep.random_transform(shaped_image)

          yield images_batch, labels_batch

    else:
      while True:
        if data_type == 'train_all':
          no_examples = len(self.train_images)
        if data_type == 'train_labeled':
          no_examples = len(self.inds_labeled)
        if data_type == 'train_unlabeled':
          no_examples = len(self.inds_unlabeled)
        elif data_type == 'val':
          no_examples = len(self.val_images)

        no_batches = int(np.ceil(float(no_examples) / params.batch_size))
        inds_batch = range(no_batches)
        if shuffle_batches:
          random.shuffle(inds_batch)

        for ind_batch in inds_batch:
          if ind_batch == no_batches - 1:
            inds = range(no_examples - params.batch_size, no_examples)
          else:
            inds = range(ind_batch * params.batch_size, (ind_batch + 1) * params.batch_size)
          
          if data_type == 'train_all':
            images_flat = self.train_images[inds]
            image_shapes = self.train_image_shapes[inds]
            prep = self.preprocessor_aug
            labels_batch = self.train_labels[inds]

          elif data_type == 'train_labeled':
            inds = [self.inds_labeled[i] for i in inds]
            inds.sort()

            images_flat = self.train_images[inds]
            image_shapes = self.train_image_shapes[inds]
            prep = self.preprocessor_aug
            labels_batch = self.train_labels[inds]

          elif data_type == 'train_unlabeled':
            inds = [self.inds_unlabeled[i] for i in inds]
            inds.sort()

            images_flat = self.train_images[inds]
            image_shapes = self.train_image_shapes[inds]
            prep = self.preprocessor_aug
            labels_batch = self.train_labels[inds]
            
          elif data_type == 'val':
            images_flat = self.val_images[inds]
            image_shapes = self.val_image_shapes[inds]
            prep = self.preprocessor
            labels_batch = self.val_labels[inds]

          images_batch = np.empty((params.batch_size, 224, 224, 3), dtype=np.float32)
          for ind_image, image_flat in enumerate(images_flat):
            shaped_image = image_flat.reshape(224, 224, 3)
            images_batch[ind_image] = prep.random_transform(shaped_image)

          yield images_batch, labels_batch