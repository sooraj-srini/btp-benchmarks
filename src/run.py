import dlgn
from data_gen import Args, data_gen_decision_tree 
import openml
import numpy as np
import dlgn, lcn, latent, tao, kernel
from sklearn.preprocessing import StandardScaler

SUITE_ID = 337
benchmark_suite = openml.study.get_suite(SUITE_ID)  

if __name__ == '__main__':
    for task_id in benchmark_suite.tasks[3:]:
        task = openml.tasks.get_task(task_id)
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


        print("DLGN performance")
        model = dlgn.trainDLGN(args)
        model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)

        print("LCN performance")
        model = lcn.trainLCN(args)
        model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)


        print("Latent Tree performance")
        model = latent.trainLatentTree(args)
        model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)


        print("TAO performance")
        model = tao.trainTAO(args)
        model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)