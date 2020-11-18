from torch.utils.data import DataLoader
from dataset import CSVDataset
from model import DRegression, Trainer
import argparse
import torch

# prepare the dataset
def prepare_data(args):
    # load the dataset
    trainval_set = CSVDataset(args, args.train_csv)
    input_dim = trainval_set.dim()
    train_set, val_set = trainval_set.data_splits()
    #data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False)

    test_set = CSVDataset(args, args.test_csv, no_target=True)
    #data loaders
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim

def main(args):
    torch.manual_seed(1)
    train_loader, val_loader, test_loader, input_dim = prepare_data(args)

    model = DRegression(input_dim)
    trainer = Trainer(model, args, train_loader, val_loader, test_loader)
    trainer.train_model()

    return

if __name__ == "__main__":
    # example command:
    # python main.py --seed 0 --train_csv train.csv --test-csv test.csv --batch-size 64 --lr 0.0005 --epoch 200 --val-set 0.1
    parser = argparse.ArgumentParser(
        description='help')
    parser.add_argument('--seed',
                        type = int,
                        default = 0,
                        metavar = 'S',
                        help = 'random seed (default: 0)')

    parser.add_argument('--train_csv',
                        default='train.csv',
                        type=str,
                        help='metafile name')

    parser.add_argument('--test_csv',
                        default='test.csv',
                        type=str,
                        help='metafile name')

    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--test-batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for testing (default: 128)')

    parser.add_argument('--lr',
                            type=float,
                            default=0.0005,
                            metavar='LR',
                            help='learning rate')

    parser.add_argument('--epoch',
                        type=int,
                        default=300,
                        metavar='E',
                        help='Number of training epochs')

    parser.add_argument('--val-set',
                            type=float,
                            default=0.1,
                            metavar='VS',
                            help='Validation set portion of the training set')

    args = parser.parse_args()

    main(args)
