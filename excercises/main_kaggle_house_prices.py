import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.nn import MSELoss,CrossEntropyLoss
from sklearn.model_selection import KFold
from operator import itemgetter

from models import LinearRegressor
from metrics import accuracy,log_rmse
from data_utils import download
from training import k_fold_training

if __name__ == '__main__':
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
    DATA_HUB = dict()
    DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
    batch_size = 64
    input_size = 330
    output_dim = 1
    lr = 5
    epochs = 10
    splits = 5
    weight_decay = 0
    see_inter_epoch_prints = False
    # Read the data from the csv files
    train_data = pd.read_csv(download('kaggle_house_train', data_hub=DATA_HUB))
    test_data = pd.read_csv(download('kaggle_house_test', data_hub=DATA_HUB))
    # Concatenate all the features in one big table 
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # If test data were inaccessible, mean and standard deviation could be
    # calculated from training data
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # After standardizing the data all means vanish, hence we can set missing
    # values to 0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
    # creates an indicator feature for it
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # Convert dataset to numpy arrays
    n_train = train_data.shape[0]
    train_features = np.array(all_features[:n_train].values, dtype=np.float32)
    test_features = np.array(all_features[n_train:].values, dtype=np.float32)
    train_labels = np.array(train_data.SalePrice.values.reshape(-1, 1), dtype=np.float32)
    # Create tuplets for the train set
    training_set = []
    for feat,lab in zip(train_features, train_labels):
        training_set.append((feat,lab))
    # Training procedure
    k_fold_training(LinearRegressor,
                    training_set,
                    {'input_dim':input_size,'output_dim':output_dim},
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    n_splits=splits)