# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, roc_curve, balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay


def get_metrics(labels, outputs, graphs_path, target_names=None, optimal_thresholds=None, plot_graphs=False):
    """
    Get classification metrics
    Args:
        - labels: target labels
        - outputs: scores obtained from the model
        - target_names: used for the classification report
        - optimal_thresholds: if defined, thresholds aren't calculed
    Returns:
        - metrics: dict with metrics
        - opt_thresholds: optimal threshols
    """

    # mAP: average using thresholds from PR curve
    #average_precision = np.mean(average_precision_score(labels, outputs, average=None))

    auc_list = []
    opt_thresholds = []
    #labels_array = np.array(labels)
    #outputs_array = np.array(outputs)
    n_classes = len(outputs[0])

    # Get AUC, optimal threshold, f1_score curve and  for each class
    #for i in range(n_classes):
    #    true_labels, pred_scores = labels_array, outputs_array[:, i]
    #    fpr, tpr, thresholds, auc = compute_auc(true_labels, pred_scores)
    #    #optimal_threshold, threshold_index = get_youden_threshold(tpr, fpr, thresholds)
    #    optimal_threshold, threshold_index = get_threshold_for_a_fpr(fpr, thresholds)
    #    f1_scores = get_f1_scores(true_labels, pred_scores, thresholds)
    #    hp_op_point, hp_f1_score = get_op_point_and_f1_score(true_labels, pred_scores, threshold=0.5)
    #    if(plot_graphs):
    #        plot_roc_and_f1_score(i, graphs_path, fpr, tpr, thresholds, auc, threshold_index, f1_scores, hp_op_point, hp_f1_score)
    #    auc_list.append(auc)
    #    opt_thresholds.append(optimal_threshold)
    
    # If there are already defined thresholds (arg) ignore the calculated ones
    #if(optimal_thresholds):
    #    opt_thresholds = optimal_thresholds
    
    # Apply thresholds
    #predicted_labels = scores_to_labels(outputs_array, opt_thresholds)
    
    predicted_labels = np.array(outputs).argmax(1)

    # Use all classes explicitly: groups may contain only one class
    class_ids = list(range(n_classes))

    # Confusion Matrix
    cnf_matrix = confusion_matrix(labels, predicted_labels, labels=class_ids)

    # Classification Report
    report = classification_report(labels, predicted_labels, labels=class_ids, target_names=target_names, output_dict=True, zero_division=0)

    # Get balanced accuracy
    balanced_accuracy = balanced_accuracy_score(labels, predicted_labels)
    report["balanced_accuracy"] = balanced_accuracy

    # Add optimal threshold and AUC to metrics dict
    #for index in range(n_classes):
    #    report[target_names[index]]['AUC'] = auc_list[index]
    #    report[target_names[index]]['Optimal_threshold'] = opt_thresholds[index]

    #metrics = [cnf_matrix, report, average_precision]
    metrics = [cnf_matrix, report]

    return metrics, opt_thresholds


def get_youden_threshold(tpr, fpr, thresholds):
    "Calculates the optimal threshold using the Younden's index"
    J = tpr - fpr
    threshold_index = np.argmax(J)
    optimal_threshold = thresholds[threshold_index]

    return optimal_threshold, threshold_index


def get_threshold_for_a_fpr(fpr, thresholds, max_fpr=0.005):
    selected_fpr = fpr[fpr <= max_fpr]
    selected_fpr = selected_fpr[np.argmax(selected_fpr)]
    threshold_index = list(fpr).index(selected_fpr)
    optimal_threshold = thresholds[threshold_index]
    
    return optimal_threshold, threshold_index


def get_op_point_and_f1_score(true_labels, pred_scores, threshold=0.5):
    """
    Using 0.5 as threshold per each class, it will be calculated the operation point
    and the f1-score.
    """
    half_prob_pred = (torch.tensor(pred_scores) > threshold).float().data.numpy()
    tn, fp, fn, tp = confusion_matrix(true_labels, half_prob_pred).ravel()
    (hp_fpr, hp_tpr) = (fp/(fp+tn), tp/(tp+fn))
    hp_f1_score = metrics.f1_score(true_labels, half_prob_pred)

    return (hp_fpr, hp_tpr), hp_f1_score


def get_f1_scores(true_labels, pred_scores, thresholds):
    f1_scores = []
    for threshold in thresholds:
        predicted_labels = (torch.tensor(pred_scores) > threshold).float().data.numpy().astype('int')
        f1_score = metrics.f1_score(true_labels, predicted_labels)
        f1_scores.append(f1_score)
    f1_scores = np.array(f1_scores)

    return f1_scores


def plot_roc_and_f1_score(class_index, graphs_path, fpr, tpr, thresholds, roc_auc, optimal_index, f1_scores, hp_point, hp_f1_score):
    
    suptitle = 'Class index: ' + str(class_index)
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, adjustable='box', aspect=1)
    fig.suptitle(suptitle, fontsize=18, fontweight='bold')

    # Plot Youden operating point
    plt.plot(fpr[optimal_index], tpr[optimal_index],'ro', label='opt threshold')

    # Plot 0.5 operating point
    plt.plot(hp_point[0], hp_point[1],'rx', label='0.5 threshold')

    # Plot ROC curve
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.legend(loc='lower right', fontsize = 16)
    ax.plot([-0.1, 1.1], [-0.1, 1.1], 'r--')
    ax.set_xlabel('False Positive Rate', fontsize = 16)
    ax.set_ylabel('True Positive Rate', fontsize = 16)
    ax.axis([-0.01, 1.01, -0.01, 1.01])

    # Plot F1-scores
    ax = fig.add_subplot(122, adjustable='box', aspect=1)
    ax.plot(thresholds, f1_scores, 'g', label='f1-score') # F1-score
    ax.plot(thresholds[optimal_index], f1_scores[optimal_index],'ro', label='opt threshold') # Optimal threshold
    ax.plot(0.5, hp_f1_score,'rx', label='0.5 threshold') # 0.5 threshold
    ax.legend(loc='lower right', fontsize = 16)
    ax.set_xlabel('Threshold', fontsize = 16)
    ax.set_ylabel('F1-score', fontsize = 16)
    ax.axis([-0.01, 1.01, -0.01, 1.01])

    plt.savefig(os.path.join(graphs_path, str(class_index)))
    plt.clf()


def compute_auc(true_labels, pred_scores):
    """
    Computes the ROC AUC
    args:
        true_labels: target labels
        pred_scores: model output
    """
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = roc_auc_score(true_labels, pred_scores)
    
    return fpr, tpr, thresholds, roc_auc


def scores_to_labels(outputs_array, thresholds):
    """
    Transform scores into labels using a different thershold per class.
    """
    predicted_labels = torch.zeros(outputs_array.shape).float()
    for i in range(outputs_array.shape[1]):
        predicted_labels[:, i] = (torch.tensor(outputs_array[:, i]) > thresholds[i]).float()
    
    return predicted_labels.data.numpy().tolist()


def plot_multilabel_confusion_matrix(cm, target_names, figure_path):
    """
    Plot multilabel CM, one CM per class
    """
    n_labels = len(target_names)
    y_subplots = n_labels // 4

    f, axes = plt.subplots(y_subplots + 1, 4, figsize=(25, 15))

    axes = axes.ravel()

    for i in range(n_labels):
        disp = ConfusionMatrixDisplay(cm[i], display_labels=[-1, i])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(target_names[i])

        if i<10:
            disp.ax_.set_xlabel('')
        if i%5!=0:
            disp.ax_.set_ylabel('')
        
        disp.im_.colorbar.remove()
    
    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(figure_path)


def plot_confusion_matrix(cm,
                          target_names,
                          figure_path,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    plt.savefig(figure_path)

    return fig