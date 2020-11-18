from pandas import read_csv
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import random_split


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, args, path, no_target=False):
        self.args = args
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        if df.empty:
            raise ValueError('The csv file is empty, check this again before training')
        # store the inputs and outputs
        if no_target:
            self.X = df.values[1:, :]
            self.y = np.zeros(len(self.X))
        else:
            self.X = df.values[1:, :-1]
            self.y = df.values[1:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')

        if not no_target:
            self.y = self.y.astype('float32')
            self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # number of rows in the dataset
    def dim(self):
        return len(self.X[0])

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def data_splits(self):
        n_val = self.args.val_set
        val_size = round(n_val * len(self.X))
        train_size = len(self.X) - val_size
        return random_split(self, [train_size, val_size])