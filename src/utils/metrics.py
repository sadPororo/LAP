""" Metric functions for the evalaution of the benchmark systems

[Metrics]
For Speaker Identification (Multi-calss Classification)
- Accuracy (ACC)
- F1-macro score (F1)
- Equal Error Rate (EER)
- the minimum of the Detection Cost Function (minDCF)

[Code Referred]
EER, minDCF: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py

"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter
from sklearn.metrics import f1_score, roc_curve


def get_Accuracy(y_target:torch.Tensor, y_pred:torch.Tensor):
    """ Get accuracy of the model prediction

    Args:
        y_target (torch.Tensor): 1D-array with class label indice in {0, C-1}, where C is the number of the classes
        y_pred (torch.Tensor): The prediction of the model, 1D-array with class label indice in {0, C-1}.

    Returns:
        the accuracy of given evaluation
    """
    assert len(y_target) == len(y_pred)
    return (y_target == y_pred).sum() / len(y_target)


def get_F1score(y_target:torch.Tensor, y_pred:torch.Tensor):
    """ Get F1-score of the model prediction

    Args:
        y_target (torch.Tensor): 1D-array with class label indice in {0, C-1}, where C is the number of the classes
        y_pred (torch.Tensor): The prediction of the model, 1D-array with class label indice in {0, C-1}.

    Returns:
        the F1 (macro) score of given evaluation
    """
    return f1_score(y_target.numpy(), y_pred.numpy(), average='macro')


#%%

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
	
	fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
	fnr = 1 - tpr
	tunedThreshold = [];
	if target_fr:
		for tfr in target_fr:
			idx = np.nanargmin(np.absolute((tfr - fnr)))
			tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	for tfa in target_fa:
		idx = np.nanargmin(np.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
		tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
	idxE = np.nanargmin(np.absolute((fnr - fpr)))
	eer  = max(fpr[idxE],fnr[idxE])*100
	
	return tunedThreshold, eer, fpr, fnr


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

