from data_loader import DataLoader
import numpy as np
import pandas as pd
import pprint


class BayesianGeneralizedRegression:

    def __init__(self, input_vector_degree, feature_vector_degree, lambda_val=2):
        self.M = input_vector_degree
        self.basis_function_degree = feature_vector_degree
        self.basis_vector_size = ((self.basis_function_degree + 1) * (self.basis_function_degree + 2)) // 2
        self.lambda_val = lambda_val
        self.output_noise_mean = 0
        self.output_noise_variance = 1
        self.y_vector = None
        self.phi_matrix = None
        self.A_matrix = None
        self.A_inv = None
        self.mse_error = None

    def get_basis_function_vector(self, x):

        basis_function_vector = np.empty(shape=(0, 0), dtype=float)
        for p in range(self.basis_function_degree + 1):
            for q in range(p + 1):
                basis_function_vector = np.append(basis_function_vector, (x[0] ** q) * (x[1] ** (p - q)))

        return basis_function_vector.reshape((self.basis_vector_size, 1))

    def compute_phi_matrix(self, dataset):
        phi_matrix = None

        for train_set_attrs, train_set_labels in dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            for i, row in train_set_attrs.iterrows():
                xi = row.values.reshape((self.M, 1))
                phi_xi = self.get_basis_function_vector(xi)
                if phi_matrix is None:
                    phi_matrix = phi_xi
                else:
                    phi_matrix = np.hstack((phi_matrix, phi_xi))

        self.phi_matrix = phi_matrix
        return self.phi_matrix

    def compute_A_matrix(self):

        phi_phi = np.matmul(self.phi_matrix, np.transpose(self.phi_matrix))
        self.A_matrix = np.add((1/self.output_noise_variance**2) * phi_phi,
                               np.linalg.inv(np.identity(self.basis_vector_size, dtype=float)))
        self.A_inv = np.linalg.inv(self.A_matrix)
        return self.A_matrix

    def prediction_mean_function(self, dataset, new_x):

        phi_y = np.matmul(self.phi_matrix, self.y_vector)

        phi_new_x = self.get_basis_function_vector(new_x)
        phi_new_x_transpose = phi_new_x.reshape((1, phi_new_x.shape[0]))

        return np.matmul((1 / self.output_noise_variance ** 2) * phi_new_x_transpose,
                         np.matmul(self.A_inv, phi_y))[0]

    def prediction_covariance_function(self, new_x):

        phi_new_x = self.get_basis_function_vector(new_x)
        phi_new_x_transpose = phi_new_x.reshape((1, phi_new_x.shape[0]))

        return np.matmul(phi_new_x_transpose,
                         np.matmul(self.A_inv, phi_new_x))

    def learn(self, dataset, report_error=False):

        self.compute_phi_matrix(dataset)
        self.compute_A_matrix()

        self.y_vector = []
        for train_set_attrs, train_set_labels in dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            self.y_vector += train_set_labels.ix[:, 0].tolist()

        self.y_vector = np.array(self.y_vector, dtype=float)
        self.y_vector.reshape((self.y_vector.shape[0], 1))

        if report_error:
            self.mse_error = self.k_fold_cross_validation(dataset)
            print('Mean Square Error = %.3f ' % self.mse_error)

    def predict_point(self, dataset, new_x):
        """
        uses mean of the posterior distribution for prediction
        """
        return self.prediction_mean_function(dataset, new_x)

    def predict(self, train_dataset, test_attrs, true_values=None):
        N = len(test_attrs)
        if not true_values.empty:
            if len(test_attrs) != len(true_values):
                raise ValueError('count mismatch in attributes and labels')
            error = 0.0

        predicted_values = []
        for i, row in test_attrs.iterrows():
            xi = row.values.reshape((self.M, 1))
            predicted_value = self.predict_point(train_dataset, xi)
            predicted_values.append(predicted_value)
            if not true_values.empty:
                true_value = true_values.iat[i, 0]
                error += (true_value - predicted_value) ** 2

        E_MSE = None
        if true_values is not None:
            E_MSE = error / N

        predicted_values = pd.DataFrame(np.array(predicted_values))
        return predicted_values, E_MSE

    def k_fold_cross_validation(self, dataset, k=10):
        cv_test_model = BayesianGeneralizedRegression(input_vector_degree=self.M,
                                                      feature_vector_degree=self.basis_function_degree)
        avg_E_MSE = 0.0
        for i in range(k):
            test_attrs, test_labels = dataset.pop(0)
            cv_test_model.learn(dataset)
            E_MSE = cv_test_model.predict(dataset, test_attrs, true_values=test_labels)[1]
            dataset.append((test_attrs, test_labels))
            avg_E_MSE += E_MSE

        avg_E_MSE = avg_E_MSE / k
        return avg_E_MSE

    def summary(self):
        print('=====Model Summary=====')
        print('Input vector size =', self.M)
        print('Basis function degree =', self.basis_function_degree)
        print('Feature vector size =', self.basis_vector_size)

        print('\nPhi Matrix of size', end=' ')
        print(self.phi_matrix.shape, ':')
        pprint.pprint(self.phi_matrix)
        print('\nA Matrix of size', end=' ')
        print(self.A_matrix.shape, ':')
        pprint.pprint(self.A_matrix)

        if self.mse_error:
            print('Mean Square Error = %.3f ' % self.mse_error)


if __name__ == '__main__':

    # Test get_basis_function_vector
    print('\n===Test get_basis_function_vector===')
    model = BayesianGeneralizedRegression(input_vector_degree=2, feature_vector_degree=2)
    result = model.get_basis_function_vector(np.array([3, 4]))
    print('phi(x) shape =', result.shape)
    print('phi(x) = ', result)

    # Test compute_phi_matrix
    print('\n===Test compute_phi_matrix===')
    model = BayesianGeneralizedRegression(input_vector_degree=2, feature_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    result = model.compute_phi_matrix([(train_attrs, train_labels)])
    print('Phi Matrix shape =', result.shape)
    print('=====Phi Matrix====')
    pprint.pprint(result)

    # Test compute_A_matrix
    print('\n===Test compute_A_matrix===')
    model = BayesianGeneralizedRegression(input_vector_degree=2, feature_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    model.compute_phi_matrix([(train_attrs, train_labels)])
    result = model.compute_A_matrix()
    print('A Matrix shape =', result.shape)
    print('=====A Matrix====')
    pprint.pprint(result)

    # Test  predict_point
    print('\n===Test predict_point===')
    model = BayesianGeneralizedRegression(input_vector_degree=2, feature_vector_degree=2)
    full_dataset = DataLoader.load_full_dataset('./regression-dataset')
    model.learn(full_dataset)
    test_new_x = np.array([7, 14]).reshape((2, 1))
    result = model.predict_point(full_dataset, test_new_x)
    print('Predicted value for', test_new_x, '=', result)

    # Test  prediction_mean_function
    print('\n===Test prediction_mean_function===')
    model = BayesianGeneralizedRegression(input_vector_degree=2, feature_vector_degree=2)
    full_dataset = DataLoader.load_full_dataset('./regression-dataset')
    model.learn(full_dataset)
    test_new_x = np.array([7, 14]).reshape((2, 1))
    result = model.prediction_mean_function(full_dataset, test_new_x)
    print('Predicted mean function value for', test_new_x, '=', result)

    # Test  prediction_covariance_function
    print('\n===Test prediction_covariance_function===')
    model = BayesianGeneralizedRegression(input_vector_degree=2, feature_vector_degree=2)
    full_dataset = DataLoader.load_full_dataset('./regression-dataset')
    model.learn(full_dataset)
    test_new_x = np.array([7, 14]).reshape((2, 1))
    result = model.prediction_covariance_function(test_new_x)
    print('Predicted covariance function value for', test_new_x, '=', result)

    # Test learn with cross validation for different values of basis function degree
    print('\n===Test learn with cross validation for different values of basis function degree===')
    for d in range(1, 5):
        model = BayesianGeneralizedRegression(input_vector_degree=2, feature_vector_degree=d)
        full_dataset = DataLoader.load_full_dataset('./regression-dataset')
        print('\nLearning in progress for basis function degree =', d)
        model.learn(full_dataset, report_error=True)
        model.summary()



