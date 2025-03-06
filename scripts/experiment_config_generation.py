import copy
import json

# Define the base configuration for each experiment

model_paths = [
    "./scripts/models_training/model_checkpoints/big_data_zero_corr_enc/dt/random_state_42max_depth_30max_features_sqrtmin_samples_leaf_1_val.pkl", # DT
    "./scripts/models_training/model_checkpoints/big_data_zero_corr_enc/knn/n_neighbors_3weights_uniformalgorithm_auton_jobs_20_val.pkl", # KNN
    "./scripts/models_training/model_checkpoints/big_data_zero_corr_enc/rf/n_estimators_100max_depth_30min_samples_split_2min_samples_leaf_1max_features_sqrtrandom_state_42n_jobs_20_val.pkl", # RF
    "./scripts/models_training\model_checkpoints/big_data_zero_corr_enc/xgboost/n_estimators_200max_depth_6learning_rate_0.1subsample_0.8colsample_bytree_0.8min_child_weight_1use_label_encoder_False_val.pkl"
]

base_experiment = {
    # "model_path": "./scripts/models_training/model_checkpoints/big_data_zero_corr_enc/knn/n_neighbors_3weights_uniformalgorithm_auton_jobs_20_val.pkl",  # Default model path
    # "model_path": [
    #     "./scripts/models_training/model_checkpoints/big_data_zero_corr_enc/dt/random_state_42max_depth_30max_features_sqrtmin_samples_leaf_1_val.pkl" # DT
    #     "./scripts/models_training/model_checkpoints/big_data_zero_corr_enc/knn/n_neighbors_3weights_uniformalgorithm_auton_jobs_20_val.pkl", # KNN
    #     "./scripts/models_training/model_checkpoints/big_data_zero_corr_enc/rf/n_estimators_100max_depth_30min_samples_split_2min_samples_leaf_1max_features_sqrtrandom_state_42n_jobs_20_val.pkl", # RF
    #     "./scripts/models_training\model_checkpoints/big_data_zero_corr_enc/xgboost/n_estimators_200max_depth_6learning_rate_0.1subsample_0.8colsample_bytree_0.8min_child_weight_1use_label_encoder_False_val.pkl" # XGBoost
    # ],
    "dataset_path": "./data/big_data_zero_corr_enc",
    "label_names": ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "Web Attack � Brute Force", "Web Attack � XSS",
                    "Web Attack � Sql Injection", "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
                    "DoS Hulk", "DoS GoldenEye", "Heartbleed", "DoS attacks-SlowHTTPTest", "DoS attacks-Hulk",
                    "Brute Force -Web", "Brute Force -XSS", "SQL Injection", "DDoS attacks-LOIC-HTTP", "Infilteration",
                    "DoS attacks-GoldenEye", "DoS attacks-Slowloris", "FTP-BruteForce", "SSH-Bruteforce", "DDOS attack-LOIC-UDP",
                    "DDOS attack-HOIC"],
    "categorical_columns_names": ["Fwd PSH Flags", "Fwd URG Flags", "FIN Flag Count", "RST Flag Count", "PSH Flag Count",
                                  "ACK Flag Count", "URG Flag Count"],
    "explainer_config": {
        "kernel_width": None,
        "kernel": None,
        "sample_around_instance": False,
        "num_features": 10,
        "num_samples": 5000,
        "sampling_func": None,
        "sampling_func_params": {}
    },
    "times_to_run": 30,
    "random_seed": None
}

# Define the specific configurations for each sampling function
sampling_configs = [

    {"sampling_func": "gaussian", "sampling_func_params": {}, "surrogate_regressor": "rf"},
    # {"sampling_func": "gamma", "sampling_func_params": {"shape_param": 1, "scale": 1}, "surrogate_regressor": "rf"},
    # {"sampling_func": "beta", "sampling_func_params": {"alpha": 0.5, "beta_param": 0.5}, "surrogate_regressor": "rf"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 3}, "surrogate_regressor": "rf"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 4}, "surrogate_regressor": "rf"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 5}, "surrogate_regressor": "rf"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 6}, "surrogate_regressor": "rf"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 7}, "surrogate_regressor": "rf"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.1}, "surrogate_regressor": "rf"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.4}, "surrogate_regressor": "rf"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.5}, "surrogate_regressor": "rf"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.7}, "surrogate_regressor": "rf"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.8}, "surrogate_regressor": "rf"},

    {"sampling_func": "gaussian", "sampling_func_params": {}, "surrogate_regressor": "dt"},
    # {"sampling_func": "gamma", "sampling_func_params": {"shape_param": 1, "scale": 1}, "surrogate_regressor": "dt"},
    # {"sampling_func": "beta", "sampling_func_params": {"alpha": 0.5, "beta_param": 0.5}, "surrogate_regressor": "dt"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 3}, "surrogate_regressor": "dt"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 4}, "surrogate_regressor": "dt"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 5}, "surrogate_regressor": "dt"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 6}, "surrogate_regressor": "dt"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 7}, "surrogate_regressor": "dt"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.1}, "surrogate_regressor": "dt"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.4}, "surrogate_regressor": "dt"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.5}, "surrogate_regressor": "dt"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.7}, "surrogate_regressor": "dt"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.8}, "surrogate_regressor": "dt"},

    {"sampling_func": "gaussian", "sampling_func_params": {}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "gamma", "sampling_func_params": {"shape_param": 1, "scale": 1}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "beta", "sampling_func_params": {"alpha": 0.5, "beta_param": 0.5}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 3}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 4}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 5}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 6}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "pareto", "sampling_func_params": {"shape_param": 7}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.1}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.4}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.5}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.7}, "surrogate_regressor": "ridge"},
    # {"sampling_func": "weibull", "sampling_func_params": {"shape_param": 0.8}, "surrogate_regressor": "ridge"},
]

# Define the list of class labels to explain for each experiment
class_labels = [0, 17, 20]

# Create the list of experiments
experiments = []
for config in sampling_configs:
    for label in class_labels:
        for model_path in model_paths:
            experiment = copy.deepcopy(base_experiment)
            experiment["model_path"] = model_path
            experiment["class_label_to_explain"] = label
            experiment["explainer_config"]["sampling_func"] = config["sampling_func"]
            experiment["explainer_config"]["sampling_func_params"] = config["sampling_func_params"]
            experiment['explainer_config']['surrogate_regressor'] = config['surrogate_regressor']
            experiments.append(experiment)

# Combine into a final config
config = {"experiments": experiments}

# Write to JSON file
with open("../experiments_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Configuration file 'experiments_config.json' generated successfully.")
