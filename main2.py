"""
Does the semi-supervised experiments.
"""
import os
import argparse
import pickle

import numpy as np
from skopt import gp_minimize
import keras
from keras.optimizers import SGD
from keras import backend as K

from resnet101 import resnet101
from multi_gpu import to_multi_gpu, to_single_gpu
import ml_loss
from datahandler import DataHandler
import metrics
import params


def update_mixed_labels(model):
    """
    Propagates labels to unlabeled examples using the features extracted with the model
    """
    if params.n_gpus > 1:
        feat_model = to_single_gpu(model)
    feat_model = keras.Model(feat_model.layers[0].input, feat_model.layers[-2].output)  # pylint: disable=E1101
    if params.n_gpus > 1:
        feat_model = to_multi_gpu(feat_model, n_gpus=params.n_gpus)

    no_batches = int(np.ceil(float(len(dh.inds_labeled)) / params.batch_size))
    l_feats = feat_model.predict_generator(dh.generator('train_labeled_sorted', aug=False, shuffle_batches=False), no_batches)
    l_feats = np.concatenate((l_feats[:(no_batches - 1) * params.batch_size], l_feats[-(len(dh.inds_labeled) - (no_batches - 1) * params.batch_size):]))

    no_batches = int(np.ceil(float(len(dh.inds_unlabeled)) / params.batch_size))
    ul_feats = feat_model.predict_generator(dh.generator('train_unlabeled', aug=False, shuffle_batches=False), no_batches)
    ul_feats = np.concatenate((ul_feats[:(no_batches - 1) * params.batch_size], ul_feats[-(len(dh.inds_unlabeled) - (no_batches - 1) * params.batch_size):]))

    min_dists = np.zeros(ul_feats.shape[0], dtype=np.float32)
    min_dist_inds = np.zeros(ul_feats.shape[0], dtype=np.int)

    for ind_unlabeled in range(ul_feats.shape[0]):
        min_dists[ind_unlabeled] = np.Inf
        for ind_labeled in range(l_feats.shape[0]):
            dist = np.linalg.norm(l_feats[ind_labeled] - ul_feats[ind_unlabeled])
            if dist < min_dists[ind_unlabeled]:
                min_dists[ind_unlabeled] = dist
                min_dist_inds[ind_unlabeled] = ind_labeled

    no_labeled = l_feats.shape[0]
    no_unlabeled = ul_feats.shape[0]
    no_mixed = no_labeled + no_unlabeled
    mean_dist = np.mean(min_dists) * (float(no_unlabeled) / no_mixed)
    similarity_scores = np.exp(-min_dists / mean_dist)

    mixed_labels = np.zeros((no_mixed, dh.no_classes[dh.dataset] + 1), dtype=np.float32)

    mixed_labels[dh.inds_labeled_sorted, :-1] = dh.train_labels[dh.inds_labeled_sorted]
    mixed_labels[dh.inds_labeled_sorted, -1] = 5

    for ind_unlabeled in range(no_unlabeled):
        mixed_labels[dh.inds_unlabeled[ind_unlabeled], :-1] = dh.train_labels[dh.inds_labeled_sorted[min_dist_inds[ind_unlabeled]]]
        mixed_labels[dh.inds_unlabeled[ind_unlabeled], -1] = similarity_scores[ind_unlabeled]

    dh.mixed_labels = mixed_labels

    """for ind in range(no_mixed):
        if ind in dh.inds_labeled:
            print('Labeled')
        else:
            print('Propagated', dh.mixed_labels[ind, -1])
        print(np.where(dh.train_labels[ind] == 1)[0])
        print(np.where(dh.mixed_labels[ind, :-1] == 1)[0])
        if ind % 5 == 0:
            print(ind)"""


def test_model(model):
    """
    Calculates the metrics on the validation data. Slightly modified for semi-supervised training
    """
    no_examples = len(dh.val_images)
    no_batches = int(np.ceil(float(no_examples) / params.batch_size))

    preds = np.empty((no_examples, dh.no_classes[args.dataset]), dtype=np.float32)
    labels = np.empty((no_examples, dh.no_classes[args.dataset]), dtype=np.float32)

    gen = dh.generator('val', aug=False, shuffle_batches=False)
    for ind_batch in range(no_batches):
        images_batch, labels_batch = gen.next()
        preds_batch = model.predict(images_batch, batch_size=params.batch_size)

        if ind_batch == no_batches - 1:
            preds[no_examples - params.batch_size:no_examples] = preds_batch[:, :-1]
            labels[no_examples - params.batch_size:no_examples] = labels_batch
        else:
            preds[ind_batch * params.batch_size:(ind_batch + 1) * params.batch_size] = preds_batch[:, :-1]
            labels[ind_batch * params.batch_size:(ind_batch + 1) * params.batch_size] = labels_batch

    return metrics.calculate_metrics(labels, preds)


def run_experiment(x):
    """
    Runs a single experiment with the given learning rate and weight decay parameters
    """
    learning_rate = x[0]
    weight_decay = x[1]

    global log_path
    if args.optimize:
        global step
        step += 1 # pylint: disable=E0602
        log_path = os.path.join('log2', args.dataset, args.ml_method, args.init, str(args.labeled_ratio), str(args.corruption_ratio), str(step))
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    with open(os.path.join(log_path, 'his.txt'), 'w') as f:
        f.write('Learning rate: ' + str(x[0]) + '\n')
        f.write('Weight decay: ' + str(x[1]) + '\n')

    K.clear_session()

    # load pretrained model for label propagation
    if args.ml_method == 'robust_warp':
        model_path = os.path.join('log', args.dataset, 'robust_warp_sup', args.init, str(args.labeled_ratio), str(args.corruption_ratio), 'best_cp.h5')
    else:
        model_path = os.path.join('log', args.dataset, args.ml_method, args.init, str(args.labeled_ratio), str(args.corruption_ratio), 'best_cp.h5')

    model_orig = resnet101(dh.no_classes[args.dataset], initialization='random', weight_decay=weight_decay)
    if params.n_gpus > 1:
        model_orig = to_multi_gpu(model_orig, n_gpus=params.n_gpus)
    model_orig.load_weights(model_path)

    update_mixed_labels(model_orig)

    model = resnet101(dh.no_classes[args.dataset] + 1, initialization='random', weight_decay=weight_decay)
    if params.n_gpus > 1:
        model_orig = to_single_gpu(model_orig)
    for ind_layer in range(len(model.layers)):
        if model.layers[ind_layer].name == model_orig.layers[ind_layer].name:
            model.layers[ind_layer].set_weights(model_orig.layers[ind_layer].get_weights())
    if params.n_gpus > 1:
        model = to_multi_gpu(model, n_gpus=params.n_gpus)

    sgd = SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss=loss_function, optimizer=sgd, metrics=[loss_function])

    ind_lr_step = 0
    train_losses = []
    val_losses = []
    patience_losses = []
    for ind_epoch in range(params.max_epoch):
        if ind_epoch % 20 == 0 and not ind_epoch == 0:
            update_mixed_labels(model)

        his = model.fit_generator(generator=dh.generator('train_mixed', aug=True),
                                  steps_per_epoch=dh.mixed_labels.shape[0] / params.batch_size,
                                  validation_data=dh.generator('val', aug=False),
                                  validation_steps=len(dh.val_images) / params.batch_size / 10,
                                  verbose=2)

        train_losses.append(his.history['loss'][0])
        val_losses.append(his.history['val_loss'][0])
        patience_losses.append(his.history['val_loss'][0])

        if min(patience_losses) == patience_losses[-1]:
            model.save_weights(os.path.join(log_path, str(ind_lr_step) + '_cp.h5'))
        elif np.argmin(np.array(patience_losses)) < len(patience_losses) - 1 - params.lr_patience:
            with open(os.path.join(log_path, 'his.txt'), 'a') as f:
                f.write('loss: ' + str(train_losses) + '\n')
                f.write('val_loss: ' + str(val_losses) + '\n')
            if ind_lr_step == params.no_lr_steps - 1:
                model.load_weights(os.path.join(log_path, str(ind_lr_step) + '_cp.h5'), by_name=True)
                break
            else:
                model.load_weights(os.path.join(log_path, str(ind_lr_step) + '_cp.h5'), by_name=True)
                learning_rate /= 10
                K.set_value(model.optimizer.lr, learning_rate)
                ind_lr_step += 1
                model.save_weights(os.path.join(log_path, str(ind_lr_step) + '_cp.h5'))
                train_losses = []
                val_losses = []

    model.save_weights(os.path.join(log_path, 'best_cp.h5'))

    res_metrics = test_model(model)
    with open(os.path.join(log_path, 'metrics.p'), 'wb') as f:
        pickle.dump(res_metrics, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(log_path, 'metrics.txt'), 'w') as f:
        f.write(str(res_metrics) + '\n')

    return -res_metrics['f1c_top3'] # negative because gp_minimize tries to minimize the result


def main():
    global log_path
    global step
    if args.optimize:
        if args.cont:
            with open(os.path.join(log_path, 'opt_res.p'), 'rb') as f:
                past_res = pickle.load(f)
            step = len(past_res['x_iters'])
            res = gp_minimize(run_experiment, params.opt_interval[args.init + '2'], n_random_starts=params.no_random_starts, n_calls=params.no_opt_iters,
                              x0=past_res['x_iters'], y0=past_res['func_vals'])
        else:
            res = gp_minimize(run_experiment, params.opt_interval[args.init], n_random_starts=params.no_random_starts, n_calls=params.no_opt_iters)

        log_path = os.path.join('log2', args.dataset, args.ml_method, args.init, str(args.labeled_ratio), str(args.corruption_ratio))
        with open(os.path.join(log_path, 'opt_res.p'), 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(log_path, 'opt_res.txt'), 'w') as f:
            f.write('Optimization results:\n')
            f.write(str(res) + '\n')
    else:
        run_experiment([params.learning_rate[args.init + '2'], params.weight_decay[args.init + '2']])


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

    log_path = os.path.join('log2', args.dataset, args.ml_method, args.init, str(args.labeled_ratio), str(args.corruption_ratio))
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

    if args.optimize:
        step = 0

    main()
