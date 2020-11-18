from torch.nn import Linear
from torch.nn import ReLU, LeakyReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn import MSELoss
import torch


# model definition
class DRegression(Module):
    # define model elements
    def __init__(self, input_dim):
        super(DRegression, self).__init__()
        self.ln1 = Linear(input_dim, 500)
        self.lk1 = LeakyReLU()
        self.ln2 = Linear(500, 200)
        self.lk2 = LeakyReLU()
        self.ln3 = Linear(200, 80)
        self.lk3 = LeakyReLU()
        self.ln4 = Linear(80, 1)

    # forward propagate input
    def forward(self, X):
        X = self.ln1(X)
        X = self.lk1(X)
        X = self.ln2(X)
        X = self.lk2(X)
        X = self.ln3(X)
        X = self.lk3(X)
        X = self.ln4(X)

        return X

class Trainer():
    def __init__(self,
                 model,
                 args,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None,
                ):

        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # define the optimization
        self.criterion = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        # self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20)

    def train_model(self):
        # enumerate epochs
        min_val_mae = 2.
        for epoch in range(self.args.epoch):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(self.train_loader):
                # clear the gradients
                self.optimizer.zero_grad()
                # compute the model output
                yhat = self.model(inputs)
                # calculate loss
                loss = self.criterion(yhat, targets)
                loss.backward()
                # update model weights
                self.optimizer.step()
                # self.lr_scheduler.step()

            print('Epoch = {}'.format(epoch))
            print('Train MAE = {}'.format(self.evaluate_model(self.train_loader)))
            val_mae = self.evaluate_model(self.val_loader)
            print('Val MAE = {}'.format(val_mae))
            if val_mae < min_val_mae:
                min_val_mae = val_mae
                self.predict_model(self.test_loader)

    # evaluate the model
    def evaluate_model(self, data_loader):
        sum_of_absolute_errors = 0.
        test_preds = []
        for i, (inputs, targets) in enumerate(data_loader):
            self.optimizer.zero_grad()
            self.model.eval()

            with torch.no_grad():
                # evaluate the model on the test set
                yhat = self.model(inputs)

                # calculate MAE
                absolute_errors = torch.abs(yhat - targets.view_as(yhat))
                sum_of_absolute_errors += torch.sum(absolute_errors)

                # retrieve array
                yhat = yhat.detach().tolist()
                test_preds.extend([y[0] for y in yhat])
                # print(test_preds)

        MAE = sum_of_absolute_errors/len(data_loader.dataset)

        return MAE

    def predict_model(self, data_loader):
        test_preds = []
        for i, (inputs, targets) in enumerate(data_loader):
            self.optimizer.zero_grad()
            self.model.eval()

            with torch.no_grad():
                # evaluate the model on the test set
                yhat = self.model(inputs)
                # retrieve array
                yhat = yhat.detach().tolist()
                test_preds.extend([y[0] for y in yhat])

        with open('test_preds.txt', 'w') as file_handle:
            file_handle.writelines("{}\n".format(item) for item in test_preds)

        return