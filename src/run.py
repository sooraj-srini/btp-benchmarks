import dlgn
from data_gen import Args, data_gen_decision_tree 
import openml
import numpy as np
import dlgn, dlgn_vt, lcn, latent, tao, kernel
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
        # print("Current task: ", count)
        # print(task)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=task.target_name)
        print(f"{dataset.name},{X.shape[0]},{X.shape[1]}")
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
        # model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)
        
        print("DLGN Value Tensor performance")
        model = dlgn.trainDLGN(args)
        # model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)

        print("LCN performance")
        model = lcn.trainLCN(args)
        # model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)


        print("Latent Tree performance")
        model = latent.trainLatentTree(args)
        # model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)


        print("TAO performance")
        model = tao.trainTAO(args)
        # model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)