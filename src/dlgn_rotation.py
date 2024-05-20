import dlgn
from data_gen import Args 
import openml
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import special_ortho_group
from xgboost import XGBClassifier

SUITE_ID = 337
benchmark_suite = openml.study.get_suite(SUITE_ID)  

if __name__ == '__main__':
    count = 0
    def get_task_size(id):
        task = openml.tasks.get_task(id)
        return task.get_X_and_y()[0].shape[1]
    arr = benchmark_suite.tasks.copy()
    arr.sort(key=lambda x: get_task_size(x))
    for task_id in arr:
        task = openml.tasks.get_task(task_id)
        count += 1
        print("Current task: ", count)
        print(task)
        dataset = task.get_dataset()
        print(f"Current Dataset:{dataset.name}")
        X, y, _, _ = dataset.get_data(target=task.target_name)
        rot = special_ortho_group.rvs(X.shape[1])
        np.random.seed(42)
        rng = np.random.permutation(X.shape[0])
        scaler = StandardScaler()

        if X.shape[0] > 100_000:
            continue

        labels = y.to_numpy()
        classes = np.unique(labels)
        labels = np.where(labels == classes[0], 0, 1)
        data_x = scaler.fit_transform(X)
        data_x, labels = data_x[rng], labels[rng]
        data_x = data_x @ rot

        num_data = len(data_x)
        num_vali = (num_data*9)//100
        num_train= (num_data*7)//10
        num_test = (num_data*21)//100
        train_data = data_x[:num_train,:]
        train_data_labels = labels[:num_train]

        vali_data = data_x[num_train:num_train+num_vali,:]
        vali_data_labels = labels[num_train:num_train+num_vali]

        test_data = data_x[num_train+num_vali :,:]
        test_data_labels = labels[num_train+num_vali :]

        args = Args()
        args.input_dim = data_x.shape[1]

        max_acc = 0
        # params = ((0.001, 5, neurons) for lr in (0.01, 0.001) for neurons in (50, 100))
        params = [(0.001, 5, 50)]
        # for param in params:
        #     lr, layers, neurons = param
        #     print(f"Learning rate {lr}, Number of layers {layers} and number of neurons {neurons}")  
        #     print("DLGN performance")
        #     args.numlayer = layers
        #     args.numnodes = neurons
        #     args.learning_rate = lr
        #     model = dlgn.trainDLGN(args)
        #     model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)
        # print(f"Maximum test accuracy for {dataset.name} is {max_acc}")
        clf = XGBClassifier(objective='binary:logistic')
        clf.fit(train_data, train_data_labels)
        acc = clf.score(test_data, test_data_labels)
        print(f"XGBoost accuracy for {dataset.name} is {acc*100}")
