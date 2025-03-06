"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  # <-- new import
from sklearn.utils import check_random_state


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None,
                 surrogate_regressor='ridge'):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            surrogate_regressor: string specifying which regressor to use by default.
                        Accepts 'ridge' (default), 'dt', or 'rf'.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.surrogate_regressor = surrogate_regressor

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_labels: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -1e8
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. See explain_instance_with_data for
           parameter explanations."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            n_method = 'forward_selection' if num_features <= 6 else 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   surrogate_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2D array. The first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. Should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation.
            num_features: maximum number of features in explanation.
            feature_selection: method for feature selection; options are:
                'forward_selection', 'highest_weights', 'lasso_path',
                'none', or 'auto'.
            surrogate_regressor: sklearn regressor to use in explanation.
                Defaults to the regressor specified in the constructor if None.
                Must support sample_weight in fit() and, if applicable,
                expose a coef_ attribute or feature_importances_.

        Returns:
            (intercept, exp, score, local_pred):
            - intercept: float.
            - exp: a sorted list of tuples (feature id, local weight),
                   sorted by decreasing absolute value of the weight.
            - score: the R^2 value of the explanation model.
            - local_pred: the prediction of the explanation model on the original instance.
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        # Choose the surrogate regressor based on the provided parameter
        if self.surrogate_regressor == 'dt':
            model_regressor = DecisionTreeRegressor(random_state=self.random_state)
        elif self.surrogate_regressor == 'rf':  # new branch for random forest
            model_regressor = RandomForestRegressor(random_state=self.random_state)
        else:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            # DecisionTreeRegressor does not have an intercept_ attribute
            intercept = getattr(easy_model, 'intercept_', 'N/A')
            print('Intercept', intercept)
            print('Prediction_local', local_pred)
            print('Right:', neighborhood_labels[0, label])
        # Attempt to obtain feature coefficients; if unavailable (as for trees) try feature importances.
        try:
            coefs = easy_model.coef_
        except AttributeError:
            if hasattr(easy_model, 'feature_importances_'):
                coefs = easy_model.feature_importances_
            else:
                coefs = np.zeros(len(used_features))
        explanation = sorted(zip(used_features, coefs),
                             key=lambda x: np.abs(x[1]), reverse=True)
        intercept_value = getattr(easy_model, 'intercept_', 0)
        return (intercept_value,
                explanation,
                prediction_score, local_pred)
