import os
import argparse
import h5py


def main():
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', choices=['nus_wide', 'ms_coco'])
  args = parser.parse_args()

  log_path = os.path.join('log', args.dataset)
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  f_dataset = h5py.File(os.path.join('dataset', args.dataset + '.h5'), 'r')
  train_images = f_dataset['train_images']
  train_image_shapes = f_dataset['train_image_shapes']
  train_labels = f_dataset['train_labels']

  if args.dataset == 'nus_wide':
    test_images = f_dataset['test_images']
    test_image_shapes = f_dataset['test_image_shapes']
    test_labels = f_dataset['test_labels']
  elif args.dataset == 'ms_coco':
    test_images = f_dataset['val_images']
    test_image_shapes = f_dataset['val_image_shapes']
    test_labels = f_dataset['val_labels']

  import pdb; pdb.set_trace()

  main()