# all adapted from https://github.com/zhufengx/SRN_multilabel


import numpy as np


def calculate_metrics(labels, preds):
  mAP = calculate_mAP(labels, preds)
  pc_top3, rc_top3, f1c_top3, po_top3, ro_top3, f1o_top3 = calculate_top3_metrics(labels, preds)

  return {'pc_top3': pc_top3, 'rc_top3': rc_top3, 'f1c_top3': f1c_top3, 
          'po_top3': po_top3, 'ro_top3': ro_top3, 'f1o_top3': f1o_top3, 'mAP': mAP}


def calculate_top3_metrics(labels, preds):
  no_examples = labels.shape[0]
  no_classes = labels.shape[1]

  top3 = np.zeros_like(preds)
  for ind_example in range(no_examples):
    top_pred_inds = np.argsort(preds[ind_example])[::-1]
    for k in range(3):
      top3[ind_example, top_pred_inds[k]] = 1

  pc_top3, rc_top3, f1c_top3 = prec_rec_f1(labels, top3)
  po_top3, ro_top3, f1o_top3 = prec_rec_f1(labels.flatten(), top3.flatten())

  return pc_top3, rc_top3, f1c_top3, po_top3, ro_top3, f1o_top3


def prec_rec_f1(labels, pred_labels):
  eps = np.finfo(np.float32).eps
  tp = labels * pred_labels
  if len(labels.shape) == 2:
    no_tp = np.sum(tp, axis=1) + eps
    no_pred = np.sum(pred_labels, axis=1) + eps
    no_pos = np.sum(labels, axis=1) + eps
  elif len(labels.shape) == 1:
    no_tp = np.sum(tp) + eps
    no_pred = np.sum(pred_labels) + eps
    no_pos = np.sum(labels) + eps

  prec_class = no_tp / no_pred + eps
  rec_class = no_tp / no_pos + eps
  f1_class = 2 * prec_class * rec_class / (prec_class + rec_class)

  return 100 * np.mean(prec_class), 100 * np.mean(rec_class), 100 * np.mean(f1_class)


def calculate_mAP(labels, preds):
  no_examples = labels.shape[0]
  no_classes = labels.shape[1]

  ap_scores = np.empty((no_classes), dtype=np.float)
  for ind_class in range(no_classes):
    ground_truth = labels[:, ind_class]
    out = preds[:, ind_class]

    sorted_inds = np.argsort(out)[::-1] # in descending order
    tp = ground_truth[sorted_inds]
    fp = 1 - ground_truth[sorted_inds]
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec = tp / np.sum(ground_truth)
    prec = tp / (fp + tp)

    rec = np.insert(rec, 0, 0)
    rec = np.append(rec, 1)
    prec = np.insert(prec, 0, 0)
    prec = np.append(prec, 0)

    for ind in range(no_examples, -1, -1):
      prec[ind] = max(prec[ind], prec[ind + 1])

    inds = np.where(rec[1:] != rec[:-1])[0] + 1
    ap_scores[ind_class] = np.sum((rec[inds] - rec[inds - 1]) * prec[inds])

  return 100 * np.mean(ap_scores)