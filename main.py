import os
import argparse
import h5py


def main():
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', choices=['nus_wide'])
  args = parser.parse_args()

  log_path = os.path.join('log', args.dataset)
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  f_dataset = h5py.File(os.path.join('dataset', args.dataset + '.h5'), 'r')

  import pdb; pdb.set_trace()

  main()