from data_gen import Args 
import openml
import numpy as np
import dlgn_vt
from sklearn.preprocessing import StandardScaler

SUITE_ID = 337
benchmark_suite = openml.study.get_suite(SUITE_ID)  

if __name__ == '__main__':
    count = 5
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
        if X.shape[0] > 100_000:
            continue
        np.random.seed(42)
        rng = np.random.permutation(X.shape[0])
        scaler = StandardScaler()

        labels = y.to_numpy()
        classes = np.unique(labels)
        labels = np.where(labels == classes[0], 0, 1)
        data_x = scaler.fit_transform(X)
        data_x, labels = data_x[rng], labels[rng]

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

        # params = ((0.001, 5, neurons) for lr in (0.01, 0.001) for neurons in (50, 100))
        params = ((C, 3, neurons) for C in (0.1, 0.06) for neurons in (20, 22))
        max_acc = 0
        for C, layers, neurons in params:
            print(f"Logistic regression {C}, Learning rate {0.001}, Number of layers {layers} and number of neurons {neurons}")  
            print("DLGN performance")
            args.numlayer = layers
            args.numnodes = neurons
            args.lr = 0.001
            args.C = C
            model = dlgn_vt.trainDLGN(args)
            acc = model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)
            max_acc = max(max_acc, acc)
        params = ((C, 4, neurons) for C in (0.03, 0.06) for neurons in (10, 12))
        for C, layers, neurons in params:
            print(f"Logistic regression {C}, Learning rate {0.001}, Number of layers {layers} and number of neurons {neurons}")  
            print("DLGN performance")
            args.numlayer = layers
            args.numnodes = neurons
            args.lr = 0.001
            args.C = C
            model = dlgn_vt.trainDLGN(args)
            acc = model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)
            max_acc = max(max_acc, acc)
        print(f"Maximum test accuracy for {dataset.name} is {max_acc}" )
