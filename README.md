Under the root directory:
There are 3 files: dataset.py contains the class for Dataset; model.py contains the designed model and the Trainer; main.py contains the main() function 
By running the following command, we can get the best model based on the lowest MAE on the validation set

python main.py --seed 0 --train_csv train.csv --test-csv test.csv --batch-size 64 --lr 0.0005 --epoch 300 --val-set 0.1
The prediction results of test samples are stored in test_preds.txt once the training is done.
