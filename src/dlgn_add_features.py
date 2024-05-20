import dlgn
from data_gen import Args 
import openml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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


        # remove unimportant features
        X = X.to_numpy()
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        # clf.fit(X, y)
        # importances = clf.feature_importances_
        # indices = np.argsort(importances)[::-1]
        for i in range(10):
            if i % 2 == 0:
                continue
            percent_features = (i + 1) * 0.1
            random_features = np.random.standard_normal(size=(X.shape[0], int(percent_features*X.shape[1]) + 1))
            X_trans = np.concatenate((X, random_features), axis=1)
            # print(indices[:int(percent_features*X.shape[1] + 1)])
            # X_trans = X[:, indices[:int(percent_features*X.shape[1] + 1)]]

            # X_trans = X[:, [2]]
            print(f"Percentage useless features is {percent_features}; Current shape of data is {X_trans.shape}")

            np.random.seed(42)
            rng = np.random.permutation(X.shape[0])
            scaler = StandardScaler()
    
            if X.shape[0] > 100_000:
                continue
    
            labels = y.to_numpy()
            classes = np.unique(labels)
            labels = np.where(labels == classes[0], 0, 1)
            data_x = scaler.fit_transform(X_trans)
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
    
            max_acc = 0
            # params = ((0.001, 5, neurons) for lr in (0.01, 0.001) for neurons in (50, 100))
            # params = ((beta, 4, neurons) for beta in (1., 10.) for neurons in (30, 150))
            params = [(1, 5, 50)]
            for beta, layers, neurons in params:
                print(f"Beta {beta}, Number of layers {layers} and number of neurons {neurons}")  
                print("DLGN performance")
                args.numlayer = layers
                args.numnodes = neurons
                # args.beta = beta
                args.lr = 0.001
                model = dlgn.trainDLGN(args)
                acc = model.train(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels)
                max_acc = max(max_acc, acc)

            clf = XGBClassifier(objective='binary:logistic')
            clf.fit(train_data, train_data_labels)
            acc = clf.score(test_data, test_data_labels)

            

            print(f"Maximum test accuracy for {dataset.name} is {max_acc}")
            print(f"Random forest classifier test accuracy score is {acc*100}")
