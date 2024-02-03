from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from data_gen import visualize
import numpy as np

class trainSVM:
    def __init__(self, args):
        self.input_dim = args.input_dim
    
    def train(self, train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=None, b_list_old=None):
        parameters = {'kernel': ['rbf'], 'C': [1000], 'gamma': [0.1]}
        svc = SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(train_data, train_data_labels)
        print(clf.best_params_)
        print(clf.best_score_)
        print("Train accuracy:", clf.score(train_data, train_data_labels))
        print("Validation accuracy:", clf.score(vali_data, vali_data_labels))
        print("Test accuracy:", clf.score(test_data, test_data_labels))


        test_pred_labels = clf.predict(test_data)  
        if self.input_dim == 2:
            visualize(test_data, test_pred_labels)
