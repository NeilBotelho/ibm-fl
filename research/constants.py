DEFAULT_CONNECTION = 'default'
DEFAULT_SERVER = 'default'

# Examples helper descriptions
GENERATE_DATA_DESC = "generates data for running IBM FL"
NUM_PARTIES_DESC = "the number of parties to split the data into"
DATASET_DESC = "which data set to use"
PATH_DESC = "directory to save the data"
PER_PARTY = "the number of data points per party"
RATIO_PER_PARTY = "Benign:Malignant Ratio per party for NON-IID data"
STRATIFY_DESC = "proportionally stratify the data according to the source distribution"

NEW_DESC = "create a new directory for this run based on current time instead of overriding"
NAME_DESC = "the name of the dataset to be generated (default is current time)"
PER_PARTY_ERR = "points per party must either specify one number of a list equal to num_parties"
RATIO_PER_PARTY_ERR = "ratio per party must either specify one number of a list equal to num_parties"
GENERATE_CONFIG_DESC = "generates aggregator and party configuration files"
PATH_CONFIG_DESC = "path to load saved config data"
MODEL_CONFIG_DESC = "which example to run"
CONNECTION_TYPE_DESC = "type of connection to use; supported types are flask, rabbitmq and websockets"
# Integration
FL_DATASETS = ["cancer"]
FL_EXAMPLES= ["keras"]
# FL_EXAMPLES = ["id3_dt", "fedavg", "keras_classifier","pfnm",
#                 "sklearn_logclassification", "sklearn_sgdclassifier",
#                 "rl_cartpole", "rl_pendulum", "coordinate_median", "krum",
#                 "naive_bayes", "keras_gradient_aggregation", "spahm", "zeno"]
FL_CONN_TYPES = ["flask"]
