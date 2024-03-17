import dlgn
from data_gen import Args, data_gen_decision_tree 
import openml
import numpy as np
import dlgn, lcn, latent, tao, kernel
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
    for task_id in arr[6:]:
        task = openml.tasks.get_task(task_id)
        count += 1
        print("Current task: ", count)
        print(task)
        dataset = task.get_dataset()
        print(f"Current Dataset:{dataset.name}")
        X, y, _, _ = dataset.get_data(target=task.target_name)
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
        params = ((lr, 3, neurons) for lr in (0.005, 0.001) for neurons in (50, 75))
        max_acc = 0
        for lr, layers, neurons in params:
            print(f"Learning rate {lr}, Number of layers {layers} and number of neurons {neurons}")  
            print("DLGN performance")
            args.numlayer = layers
            args.numnodes = neurons
            args.lr = lr
            model = dlgn.trainDLGN(args)
            acc = model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)
            max_acc = max(max_acc, acc)
        print(f"Maximum test accuracy for {dataset.name} is {max_acc}")
