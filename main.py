import os
import argparse
import pickle

import numpy as np
from skopt import gp_minimize
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from resnet101 import resnet101
from multi_gpu import to_multi_gpu
from lossplotter import LossPlotter
import ml_loss
from datahandler import DataHandler
import metrics
import params


def test_model(model):
  no_examples = len(dh.val_images)
  no_batches = int(np.ceil(float(no_examples) / params.batch_size))

  preds = np.empty((no_examples, dh.no_classes[args.dataset]), dtype=np.float32)
  labels = np.empty((no_examples, dh.no_classes[args.dataset]), dtype=np.float32)

  gen = dh.generator('val', shuffle_batches=False)
  for ind_batch in range(no_batches):
    images_batch, labels_batch = gen.next()
    preds_batch = model.predict(images_batch, batch_size=params.batch_size)

    if ind_batch == no_batches - 1:
      preds[no_examples - params.batch_size:no_examples] = preds_batch
      labels[no_examples - params.batch_size:no_examples] = labels_batch
    else:
      preds[ind_batch * params.batch_size:(ind_batch + 1) * params.batch_size] = preds_batch
      labels[ind_batch * params.batch_size:(ind_batch + 1) * params.batch_size] = labels_batch

  return metrics.calculate_metrics(labels, preds)


def run_experiment(x):
  learning_rate = x[0]
  weight_decay = x[1]

  global log_path
  if args.optimize:
    global step
    step += 1
    log_path = os.path.join('log', args.dataset, args.ml_method, args.init, str(args.labeled_ratio), str(args.corruption_ratio), str(step))
    if not os.path.exists(log_path):
      os.makedirs(log_path)

  with open(os.path.join(log_path, 'his.txt'), 'a') as f:
    f.write('Learning rate: ' + str(x[0]) + '\n')
    f.write('Weight decay: ' + str(x[1]) + '\n')

  K.clear_session()
  model = resnet101(dh.no_classes[args.dataset], initialization=args.init, weight_decay=weight_decay)
  if params.n_gpus > 1:
    model = to_multi_gpu(model, n_gpus=params.n_gpus)

  sgd = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
  model.compile(loss=loss_function, optimizer=sgd, metrics=[loss_function])

  ind_lr_step = 0
  lowest_loss = np.inf
  while ind_lr_step < params.no_lr_steps:
    model_checkpoint= ModelCheckpoint(os.path.join(log_path, str(ind_lr_step) + '_cp.h5'), monitor='val_loss', save_best_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', patience=params.lr_patience)
    loss_plotter = LossPlotter(os.path.join(log_path, str(ind_lr_step) + '_losses.png'))

    his = model.fit_generator(generator=dh.generator('train_labeled'),
                              steps_per_epoch=len(dh.inds_labeled) / params.batch_size,
                              epochs=params.max_epoch,
                              callbacks=[model_checkpoint, early_stopper, loss_plotter],
                              validation_data=dh.generator('val'),
                              validation_steps=len(dh.val_images) / params.batch_size / 10,
                              verbose=2)
    with open(os.path.join(log_path, str(ind_lr_step) + '_his.p'), 'wb') as f:
      pickle.dump(his.history, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(log_path, 'his.txt'), 'a') as f:
      f.write('Training history:\n')
      f.write(str(his.history) + '\n')

    if np.min(his.history['val_loss']) < lowest_loss:
      lowest_loss = np.min(his.history['val_loss'])
    else:
      model.load_weights(os.path.join(log_path, str(ind_lr_step - 1) + '_cp.h5'), by_name=True)
      break

    model.load_weights(os.path.join(log_path, str(ind_lr_step) + '_cp.h5'), by_name=True)
    learning_rate /= 10
    K.set_value(model.optimizer.lr, learning_rate)
    ind_lr_step += 1

  model.save_weights(os.path.join(log_path, 'best_cp.h5'))

  metrics = test_model(model)
  with open(os.path.join(log_path, 'metrics.p'), 'wb') as f:
    pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
  with open(os.path.join(log_path, 'metrics.txt'), 'w') as f:
    f.write(str(metrics) + '\n')
  
  return -metrics['f1c_top3'] # negative because gp_minimize tries to minimize the result


def main():
  global log_path
  global step
  if args.optimize:
    if args.cont:
      with open(os.path.join(log_path, 'opt_res.p'), 'rb') as f:
        past_res = pickle.load(f)
      step = len(past_res['x_iters'])
      res = gp_minimize(run_experiment, params.opt_interval[args.init], n_random_starts=params.no_random_starts, n_calls=params.no_opt_iters,
                        x0=past_res['x_iters'], y0=past_res['func_vals'])
    else:
      res = gp_minimize(run_experiment, params.opt_interval[args.init], n_random_starts=params.no_random_starts, n_calls=params.no_opt_iters)
 
    log_path = os.path.join('log', args.dataset, args.ml_method, args.init, str(args.labeled_ratio), str(args.corruption_ratio))
    with open(os.path.join(log_path, 'opt_res.p'), 'wb') as f:
      pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(log_path, 'opt_res.txt'), 'w') as f:
      f.write('Optimization results:\n')
      f.write(str(res) + '\n')
  else:
    run_experiment([params.learning_rate[args.init], params.weight_decay[args.init]])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', choices=['nus_wide', 'ms_coco'])
  parser.add_argument('init', choices=['imagenet', 'random'])
  parser.add_argument('ml_method', choices=['br', 'sm', 'pwr', 'warp', 'robust_warp', 'robust_warp_sup'])
  parser.add_argument('labeled_ratio', type=int, choices=[10, 20, 100]) # percentage
  parser.add_argument('corruption_ratio', type=int, choices=range(0, 60, 10)) # percentage
  parser.add_argument('--optimize', action='store_true', help='does hyperparameter optimization')
  parser.add_argument('--cont', action='store_true', help='continues optimization')
  args = parser.parse_args()

  log_path = os.path.join('log', args.dataset, args.ml_method, args.init, str(args.labeled_ratio), str(args.corruption_ratio))
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  dh = DataHandler(args.dataset, args.labeled_ratio, args.corruption_ratio)

  if args.ml_method == 'br':
    loss_function = ml_loss.binary_relevance.binary_relevance
  elif args.ml_method == 'sm':
    loss_function = ml_loss.ml_softmax.ml_softmax
  elif args.ml_method == 'pwr':
    loss_function = ml_loss.pairwise_ranking.pairwise_ranking
  elif args.ml_method == 'warp':
    loss_function = ml_loss.warp_py.warp
  elif args.ml_method == 'robust_warp':
    loss_function = ml_loss.robust_warp_py.robust_warp
  elif args.ml_method == 'robust_warp_sup':
    #robust_warp_sup is robust_warp that can only be used for fully-supervised training
    loss_function = ml_loss.robust_warp_sup_py.robust_warp_sup

  if args.optimize:
    step = 0

  main()