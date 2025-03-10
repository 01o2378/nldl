from dataclasses import dataclass

from lime_experiment.explained_model import ExplainedModel
from lime_experiment.experiment_data import ExperimentData


@dataclass
class LimeExplainerConfig:
    kernel_width: int
    kernel: callable
    sample_around_instance: bool
    num_features: int
    num_samples: int
    sampling_func: str
    sampling_func_params: dict
    surrogate_regressor: str

    """
    In these experiments, the following explainer constants will be maintained: the model_regressor will default to Ridge 
    regression. The explanation will be generated for all labels of the model. The 
    Euclidean distance metric will be used as the default for calculating proximity in feature space, which impacts 
    how the local models are weighted. Additionally, the feature selection method will default to 'auto', 
    allowing LIME to automatically choose the best features based on criteria like forward selection or Lasso, 
    depending on the data and model.
    """

    def as_records_dict(self):
        return {
            "kernel_width": self.kernel_width if self.kernel_width is not None else "default (sqrt(num_features) * 0.75)",
            "kernel": self.kernel if self.kernel is not None else "default (exponential)",
            "sample_around_instance": self.sample_around_instance,
            "num_features": self.num_features,
            "num_samples": self.num_samples,
            "sampling_func": self.sampling_func,
            "sampling_func_params": self.sampling_func_params,
            "surrogate_regressor": self.surrogate_regressor,
        }


@dataclass
class LimeExperimentConfig:
    explained_model: ExplainedModel
    experiment_data: ExperimentData
    explainer_config: LimeExplainerConfig
    times_to_run: int
    random_seed: int = None
    save_results: bool = True
    save_explanations: bool = True
    mode: str = "classification"
    class_label_to_test: int = None # If None, a test instance will be randomly selected from the dataset.
    # If a class label is provided, a random instance of that class will be selected.

    def as_records_dict(self):
        configuration_records_dict = {
            "times_explained": self.times_to_run,
            "random_seed": self.random_seed,
            "dataset": self.experiment_data.get_dataset_name(),
            "number_of_features": self.experiment_data.get_feature_count(),
            "number_of_categorical_features": self.experiment_data.get_categorical_features_count(),
            "categorical_features": self.experiment_data.get_categorical_features_names(),
            "explained_model": self.explained_model.model_path,
            "explained_model_type": self.explained_model.model_type,
        }
        configuration_records_dict.update(self.explainer_config.as_records_dict())
        return configuration_records_dict

@dataclass
class LabelExplanationMetrics:
    """
    A dataclass to hold the evaluation metrics for explanations of a specific label.

    Attributes:
    fidelity (float): The average fidelity score (e.g., average R² score) of the model
                      across explanation runs for a specific label. Fidelity measures how well
                      the explanations reflect the underlying model's behavior for the label.

    stability (float): The average stability score (e.g., average Jaccard similarity)
                       across explanation runs for a specific label. Stability measures how
                       consistent the explanations are across multiple runs or different
                       perturbations for the label.
    """
    fidelity: float
    stability: float
