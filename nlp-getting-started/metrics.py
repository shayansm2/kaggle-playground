from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay, f1_score
from matplotlib import pyplot as plt

class MetricsCalculator(object):
    def __init__(self, y_actual, y_probabilities):
        self.y_true = y_actual
        self.y_pred = y_probabilities
        self.threshold = 0.5

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_auc(self):
        return roc_auc_score(self.y_true, self.y_pred)

    def get_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred >= self.threshold)

    def get_recall(self):
        return recall_score(self.y_true, self.y_pred >= self.threshold)

    def get_precision(self):
        return precision_score(self.y_true, self.y_pred >= self.threshold)

    def get_f1(self):
        return f1_score(self.y_true, self.y_pred >= self.threshold)

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred >= self.threshold)

    def show_confusion_matrix(self, axis=None):
        confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix(self.y_true, (self.y_pred > self.threshold)))
        confusion_matrix_display.plot(cmap='magma', ax=axis)

    def show_roc_curve(self, axis=None):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred)
        if axis is None:
            axis = plt
        axis.plot(fpr, tpr)
        axis.plot([0, 1], [0, 1])
