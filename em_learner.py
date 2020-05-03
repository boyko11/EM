import numpy as np
from service.report_service import ReportService


class EMLearner:

    def __init__(self):
        self.report_service = ReportService()
        self.trained_centroids = None
        self.k = None
        self.current_priors = None
        self.min_probability = 0.0001  # avoid divide by zero errors
        self.current_means = None
        self.current_covariance_matrices = None

    @staticmethod
    def calculate_probability_of_data(feature_data, this_mean, this_covariance_matrix):
        # part1 = 1.0 / ((2 * np.pi) * np.linalg.det(this_covariance_matrix))
        part1 = 1.0 / (((2 * np.pi) ** (feature_data.shape[1] / 2.0)) * np.linalg.det(this_covariance_matrix) ** 0.5)
        record_minus_means = feature_data - this_mean
        part2 = np.exp(-0.5 * np.dot(np.dot(record_minus_means, np.linalg.inv(this_covariance_matrix)),
                                     record_minus_means.T))

        return (part1 * part2).diagonal()

    def train(self, data=None, labels=None, k=2):

        self.k = k
        self.current_priors = [1 / k] * k
        self.current_means = self.generate_k_random_gaussians(data, labels, k)
        self.current_covariance_matrices = [np.identity(data.shape[1]), np.identity(data.shape[1])]

        iter_count = 1
        cost_history = []
        converged = False

        while not converged:

            # calculate probabilities records came from each of the k gaussians

            record_k_gaussian_probabilites, _ = self.estimate_probabilities(data)

            # reestimate the k new means means
            record_k_gaussian_probabilites_sum = np.sum(record_k_gaussian_probabilites, axis=0)

            weights = record_k_gaussian_probabilites / record_k_gaussian_probabilites_sum

            new_means = self.current_means.copy()
            for k_index in range(self.k):
                records_multiplied_by_weights = data * weights[:, k_index].reshape(weights.shape[0], 1)
                records_multiplied_by_weights_sum = np.sum(records_multiplied_by_weights, axis=0)
                new_means[k_index, :] = records_multiplied_by_weights_sum

            # re-estimate the priors
            weights_sum = np.sum(weights, axis=0)
            self.current_priors = weights_sum / np.sum(weights_sum)

            # re-estimate the k covariance matrices

            new_covariance_matrices = self.current_covariance_matrices.copy()
            for k_index in range(self.k):
                new_covariance_matrix_for_this_k = np.zeros(shape=self.current_covariance_matrices[k_index].shape)
                for record_index, data_row in enumerate(data):
                    diff = (data_row - self.current_means[k_index]).reshape(data.shape[1], 1)
                    product = weights[record_index, k_index] * diff * diff.T
                    new_covariance_matrix_for_this_k += product
                new_covariance_matrices[k_index] = new_covariance_matrix_for_this_k

            self.current_covariance_matrices = new_covariance_matrices

            self.current_means = new_means

            cluster_assignments = np.argmax(record_k_gaussian_probabilites, axis=1)
            if iter_count > 1:
                cost = self.calculate_cost(cluster_assignments, labels)
                cost_history.append(cost)

            if len(cost_history) > 3 and cost_history[-1] == cost_history[-2] == cost_history[-3]:
                print("Converged at cost ", cost_history[-1], " in ", iter_count, " iterations.")
                print("Cost History: ", cost_history)
                converged = True

            iter_count += 1

    def estimate_probabilities(self, data):

        record_k_gaussian_probabilites = np.empty(shape=(data.shape[0], self.k))

        for k_index, current_mean in enumerate(self.current_means):
            record_probability_given_current_gaussian = \
                self.calculate_probability_of_data(data, current_mean, self.current_covariance_matrices[k_index])
            record_k_gaussian_probabilites[:, k_index] = record_probability_given_current_gaussian

        record_k_gaussian_probabilites[record_k_gaussian_probabilites < self.min_probability] = self.min_probability
        probabilities_sums = np.sum(record_k_gaussian_probabilites * self.current_priors, axis=1)

        # normalize probabilities
        record_k_gaussian_probabilites = np.divide(record_k_gaussian_probabilites * self.current_priors,
                                                   probabilities_sums.reshape((probabilities_sums.size, 1)))

        hard_cluster_assignments = np.argmax(record_k_gaussian_probabilites, axis=1)
        return record_k_gaussian_probabilites, hard_cluster_assignments


    @staticmethod
    def calculate_cost(predictions, labels):
        return np.sum(np.abs(np.array(predictions) - labels))

    @staticmethod
    def generate_k_random_gaussians(data, labels, k):

        means = np.mean(data, axis=0)

        k_random_means = np.empty(shape=(k, data.shape[1]))
        st_devs = np.std(data, axis=0)

        for k_index in range(k):
            k_th_random_mean = []
            k_th_random_st_dev = []
            for feature_index in range(means.size):
                random_number_for_this_dimensions = np.random.normal(means[feature_index], st_devs[feature_index], 1)[0]
                k_th_random_mean.append(random_number_for_this_dimensions)
                k_th_random_st_dev.append(np.random.uniform(low=np.min(st_devs), high=np.max(st_devs)))

            k_random_means[k_index, :] = k_th_random_mean

        k_random_means[1, :] = data[-2, :]
        k_random_means[0, :] = data[-1, :]

        # zero_class_data = data[np.where(labels == 0)[0]]
        # random_zero_label_record = np.random.randint(zero_class_data.shape[0])
        # k_random_means[0, :] = zero_class_data[random_zero_label_record, :]
        # one_class_data = data[np.where(labels == 1)[0]]
        # random_one_label_record = np.random.randint(one_class_data.shape[0])
        # k_random_means[1, :] = one_class_data[random_one_label_record, :]
        return k_random_means



