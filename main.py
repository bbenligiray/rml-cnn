import os
import argparse
import h5py

from resnet101 import resnet101
from multi_gpu import to_multi_gpu


def load_pretrained_model(weight_decay):
  if args.dataset == 'nus_wide':
    no_classes = 81
  elif args.dataset == 'ms_coco':
    no_classes = 80
  if args.ml_method == 'binary_relevance':
    final_activation = 'softmax'
  else:
    final_activation = None
  return resnet101(no_classes, final_activation, weight_decay=weight_decay)


def main():
  model = load_pretrained_model(weight_decay=1E43)
  model = to_multi_gpu(model, n_gpus=4)
  import pdb; pdb.set_trace()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', choices=['nus_wide', 'ms_coco'])
  parser.add_argument('ml_method', choices=['binary_relevance'])
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

  main()