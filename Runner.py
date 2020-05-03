from service.data_service import DataService
from em_learner import EMLearner
from service.report_service import ReportService


class Runner:

    def __init__(self, normalization_method='z'):
        self.em_learner = None
        self.normalization_method = normalization_method
        self.report_service = ReportService()

    def run(self, k=2):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        labels = data[:, 1]

        self.em_learner = EMLearner()

        normalized_data = DataService.normalize(data, method=self.normalization_method)
        normalized_feature_data = normalized_data[:, 2:]

        feature_data = data[:, 2:]
        # feature_data = normalized_feature_data
        self.em_learner.train(feature_data, labels, k=k)

        record_k_gaussian_probabilites, cluster_assignments = self.em_learner.estimate_probabilities(feature_data)
        self.report_service.accuracy_stats_report(data, record_k_gaussian_probabilites, cluster_assignments, labels)
        self.report_service.report(cluster_assignments, labels, self.em_learner.k)


if __name__ == "__main__":

    Runner(normalization_method='min-max').run(k=2)
