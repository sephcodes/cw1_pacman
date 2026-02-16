# classifier.py
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import VotingClassifier

from sklearn.calibration import CalibratedClassifierCV


param_grid = {
    "learning_rate": [0.001, 0.005, 0.01],
    "num_epochs": [10, 50, 100],
    "regularization": [0.001, 0.01, 0.1]
}

# -------------------------------------------------
# Logistic Regression Softmax with Gradient Descent
# -------------------------------------------------
class SoftmaxGDClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, learning_rate=0.01, num_epochs=100, regularization=0.01):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.regularization = regularization

    def _softmax(self, x):
        # softmax from https://en.wikipedia.org/wiki/Multinomial_logistic_regression and 8.3.7 Murphy book
        # but added bias term and vectorized
        if x.ndim == 1:
            z = self.w_ @ x + self.b_
            return np.exp(z) / np.sum(np.exp(z))
        else:
            z = x @ self.w_.T + self.b_
            return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def _predict_loss_grad(self, x, y):
        probs = self._softmax(x)

        # https://en.wikipedia.org/wiki/Multinomial_logistic_regression#Likelihood_function
        loss = -np.log(probs[y])

        # (4.109) PRML book (https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
        t = np.zeros_like(probs)
        t[y] = 1
        grad_w = np.outer(probs - t, x)
        grad_b = probs - t

        return loss, grad_w, grad_b
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        N = X.shape[0] # number of samples
        # adding self.classes_ for gridsearch
        self.classes_ = np.unique(y)
        K = len(self.classes_) # number of classes
        D = X.shape[1] # feature space
        self.w_ = np.zeros((K, D)) # weights
        self.b_ = np.zeros(K) # biases

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            grad_w_sum = np.zeros_like(self.w_)
            grad_b_sum = np.zeros_like(self.b_)

            for x, y_i in zip(X, y):
                loss, grad_w, grad_b = self._predict_loss_grad(x, y_i)
                total_loss += loss
                grad_w_sum += grad_w
                grad_b_sum += grad_b

            # batch gradient descent
            grad_w_sum += self.regularization * self.w_ # adding regularization term to the gradient sum
            self.w_ -= self.learning_rate * grad_w_sum / N
            self.b_ -= self.learning_rate * grad_b_sum / N

        return self
    
    def predict_proba(self, X):
        X = np.atleast_2d(X)
        return self._softmax(X)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    


# ---------------------------------
# Sklearn Neural Network Classifier
# ---------------------------------
# https://sklearn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
class NeuralNetworkClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.hidden_dim = 8
        self.learning_rate_init = 0.01
        self.batch_size = 10
        self.epochs = 100
        self.weight_decay = 0.01

    def fit(self, X, y):
        self.model_ = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim,),
            alpha=self.weight_decay, # L2 regularization https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            max_iter=self.epochs,
            shuffle=True,
            verbose=True
        )
        # neural net is dominating the ensemble predicted probabilities with overconfident predictions, so calibrating
        # https://scikit-learn.org/stable/modules/calibration.html#probability-calibration
        self.model_ = CalibratedClassifierCV(
            self.model_,
            cv=5
        )

        self.model_.fit(X, y)
        # training accuracy
        y_pred = self.model_.predict(X)
        train_acc = np.mean(np.array(y_pred) == y)
        print(f"Train Accuracy: {100 * train_acc}%")
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)
    

# -------------------------
# Support Vector Classifier
# -------------------------
# https://sklearn.org/stable/modules/generated/sklearn.svm.SVC.html
class SupportVectorClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, gamma="auto"):
        self.gamma = gamma
        
    def fit(self, X, y):
        self.model_ = SVC(gamma=self.gamma, probability=True)
        # same calibration here
        self.model_ = CalibratedClassifierCV(
            self.model_,
            cv=5
        )
        self.model_.fit(X, y)
        # training accuracy
        y_pred = self.model_.predict(X)
        train_acc = np.mean(np.array(y_pred) == y)
        print(f"Train Accuracy: {100 * train_acc}%")
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def predict(self, X):
        return self.model_.predict(X)
    

# --------------------------------
# Bernoulli Naive Bayes Classifier
# --------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
# "Like MultinomialNB, this classifier is suitable for discrete data. 
# The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features."
# CW FAQ said NB is considered a simple model but based on above quote, it seems suitable for this experiment
class BernoulliNBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model_ = BernoulliNB()
        self.model_.fit(X, y)
        # training accuracy
        y_pred = self.model_.predict(X)
        train_acc = np.mean(np.array(y_pred) == y)
        print(f"Train Accuracy: {100 * train_acc}%")
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def predict(self, X):
        return self.model_.predict(X)



class Classifier:
    def __init__(self):
        pass

    def reset(self):
        pass
    
    def fit(self, data, target):
        clf = SoftmaxGDClassifier()
        self.gs = GridSearchCV(clf, param_grid, scoring="accuracy", cv=5, n_jobs=-1)

        self.nn = NeuralNetworkClassifier()

        self.svc = SupportVectorClassifier()

        self.nb = BernoulliNBClassifier()
        
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
        self.eclf = VotingClassifier(
            estimators=[('lr', self.gs), ('nn', self.nn), ('svc', self.svc), ('nb', self.nb)], 
            voting='soft'
        )
        self.eclf.fit(data, target)
        return self

    def predict(self, data, legal=None):
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        print("INPUT: ", data)
        
        # doing this to see the individual probabilities for each model
        results = []
        ensemble_proba = np.round(self.eclf.predict_proba(data), 3)
        ensemble_pred = self.eclf.predict(data)
        for i in range(len(data)):
            sample_info = {
                "ensemble_proba": ensemble_proba[i],
                "ensemble_prediction": ensemble_pred[i],
                "individual_probas": {}
            }
            for name, est in zip(self.eclf.named_estimators_.keys(), self.eclf.estimators_):
                sample_info["individual_probas"][name] = np.round(est.predict_proba(
                    data[i].reshape(1, -1)
                )[0], 3)
            results.append(sample_info)
        
        print(results)
        return ensemble_pred
