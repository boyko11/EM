import numpy as np
from service.plot_service import PlotService


class ReportService:

    def __init__(self):
        pass

    @staticmethod
    def report(cluster_assignments, labels_data, k):

        plot_data = {}

        for cluster_index in range(k):
            cluster_0_assignments_indices = np.where(cluster_assignments == cluster_index)
            data_records_assigned_to_cluster = labels_data[cluster_0_assignments_indices]
            unique_labels, counts_per_label = np.unique(data_records_assigned_to_cluster, return_counts=True)
            plot_data[cluster_index] = counts_per_label

        PlotService().plot_clusters_per_label_barchart(('Benign', 'Malignant'), k, plot_data)

    def accuracy_stats_report(self, data, prediction_probabilities, predictions, labels_data):

        predictions = np.array(predictions)

        accuracy = 1 - np.sum(np.abs(predictions - labels_data)) / labels_data.shape[0]

        print("Accuracy: ", accuracy)

        positive_labels_count = np.count_nonzero(labels_data)
        negative_labels_count = labels_data.shape[0] - positive_labels_count
        positive_predictions_count = np.count_nonzero(predictions)
        negative_predictions_count = labels_data.shape[0] - positive_predictions_count

        print("Positive Labels, Positive Predictions: ", positive_labels_count, positive_predictions_count)
        print("Negative Labels, Negative Predictions: ", negative_labels_count, negative_predictions_count)

        labels_for_class1_predictions = labels_data[predictions == 1]
        true_positives_class1 = np.count_nonzero(labels_for_class1_predictions)
        false_negatives_class0 = labels_for_class1_predictions.shape[0] - true_positives_class1

        labels_for_class0_predictions = labels_data[predictions == 0]
        false_negatives_class1 = np.count_nonzero(labels_for_class0_predictions)
        true_positives_class0 = labels_for_class0_predictions.shape[0] - false_negatives_class1

        print('Class 1, true_positives, false_positives: ', true_positives_class1,
              positive_predictions_count - true_positives_class1)
        precision_class1 = np.around(true_positives_class1 / positive_predictions_count, 3)
        recall_class1 = np.around(true_positives_class1 / (true_positives_class1 + false_negatives_class1), 3)
        class1_f1_score = np.around(2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1), 3)

        print('Class 0, true_positives, false_positives: ', true_positives_class0,
              negative_predictions_count - true_positives_class0)
        precision_class0 = np.around(true_positives_class0 / negative_predictions_count, 3)
        recall_class0 = np.around(true_positives_class0 / (true_positives_class0 + false_negatives_class0), 3)
        class0_f1_score = np.around(2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0), 3)

        print('precision class1: ', precision_class1)
        print('recall class1: ', recall_class1)
        print('f1 score class1: ', class1_f1_score)
        print('precision class0: ', precision_class0)
        print('recall class0: ', recall_class0)
        print('f1 score class0: ', class0_f1_score)

        self.print_error_stats(data, labels_data, prediction_probabilities, predictions)

    @staticmethod
    def print_error_stats(data, labels_data, prediction_probabilities, predictions):
        record_ids = data[:, 0].flatten()
        np.set_printoptions(suppress=True)
        # | Record ID | Label | Probability 0 | Probability 1 | Rounded Error |
        for i in range(labels_data.shape[0]):
            record_id = record_ids[i]
            label = 'Malignant' if labels_data[i] == 1 else 'Benign'
            prediction_error = np.abs(labels_data[i] - predictions[i])

            print('|{0}|{1}|{2}|{3}|{4}|'.format(int(record_id), label,
                                                 np.around(prediction_probabilities[i, 0], decimals=3),
                                                 np.around(prediction_probabilities[i, 1], decimals=3), prediction_error))
