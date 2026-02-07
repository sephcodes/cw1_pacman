# classifier.py
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

# import torch
# from torch import nn
# from torch.utils.data import TensorDataset, DataLoader

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier


param_grid = {
    "learning_rate": [0.001, 0.005, 0.01],
    "num_epochs": [10, 50, 100]
}

# -------------------------------------------------
# Logistic Regression Softmax with Gradient Descent
# -------------------------------------------------
class SoftmaxGDClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def _softmax(self, x):
        # softmax from https://en.wikipedia.org/wiki/Multinomial_logistic_regression and 8.3.7 Murphy book
        # but added bias term like w0 and vectorized
        if x.ndim == 1:
            z = self.w_ @ x + self.b_
            # z -= np.max(z)
            return np.exp(z) / np.sum(np.exp(z))
        else:
            z = x @ self.w_.T + self.b_
            # z -= np.max(z, axis=1, keepdims=True)
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
            self.w_ -= self.learning_rate * grad_w_sum / N
            self.b_ -= self.learning_rate * grad_b_sum / N

        return self
    
    def predict_proba(self, X):
        X = np.atleast_2d(X)
        return self._softmax(X)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    

# ---------------------------------
# Pytorch Neural Network Classifier
# ---------------------------------
# https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.linear_relu_stack = nn.Sequential(
#             # Small NN model. Any bigger was massively overfitting
#             # Will add dropout if still overfitting
#             nn.Linear(self.input_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.output_dim)
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits

# class NeuralNetworkClassifier(ClassifierMixin, BaseEstimator):
#     def __init__(self):
#         self.input_dim = 25
#         self.output_dim = 4
#         self.hidden_dim = 8
#         self.learning_rate = 0.01
#         self.batch_size = 10
#         self.epochs = 100
#         self.loss_fn = nn.CrossEntropyLoss()

#         # # Model creation
#         # self.model = NeuralNetwork(input_dim=self.input_dim, output_dim=self.output_dim, hidden_dim=self.hidden_dim)
#         # # Optimizer
#         # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
    
#     # https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
#     def _train_loop(self, dataloader):
#         self.model_.train()
#         # Set the model to training mode - important for batch normalization and dropout layers
#         # Unnecessary in this situation but added for best practices
#         size = len(dataloader.dataset)
#         for batch, (X, y) in enumerate(dataloader):
#             # Compute prediction and loss
#             pred = self.model_(X)
#             loss = self.loss_fn(pred, y)

#             # Backpropagation
#             loss.backward()
#             self.optimizer_.step()
#             self.optimizer_.zero_grad()

#             if batch == 0:
#                 loss, current = loss.item(), batch * self.batch_size + len(X)
#                 print(f"loss: {loss:>5f}")

#         # Added train eval to monitor if train acc much higher than test acc (overfitting)
#         # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
#         # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
#         self.model_.eval()
#         correct = 0
#         with torch.no_grad():
#             for X, y in dataloader:
#                 pred = self.model_(X)
#                 correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#         correct /= size
#         print(f"Train Accuracy: {(100*correct):>0.1f}")
    
#     def fit(self, X, y):
        
#         # Moving Model creation to fit instead of __init__ because of sklearn ensemble
#         self.model_ = NeuralNetwork(input_dim=self.input_dim, output_dim=self.output_dim, hidden_dim=self.hidden_dim)
#         self.optimizer_ = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=0.01)

#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.long)
#         dataset = TensorDataset(X_tensor, y_tensor)
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(self.epochs):
#             print(f"Epoch {epoch+1}\n-------------------------------")
#             self._train_loop(dataloader)

#     def predict_proba(self, X):
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         if X_tensor.dim() == 1:
#             X_tensor = X_tensor.unsqueeze(0)
#         self.model_.eval()
#         with torch.no_grad():
#             pred_log = self.model_(X_tensor)
#             pred_proba = torch.softmax(pred_log, dim=1)
#         return pred_proba.numpy()
    
#     def predict(self, X):
#         pred = self.predict_proba(X).argmax(1)
#         return pred


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
    

# https://sklearn.org/stable/modules/generated/sklearn.svm.SVC.html
class SupportVectorClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, gamma):
        self.gamma = gamma
        
    def fit(self, X, y):
        self.model_ = SVC(gamma=self.gamma, probability=True)
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
        self.svc = SupportVectorClassifier(gamma="auto")
        
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
        self.eclf = VotingClassifier(
            estimators=[('lr', self.gs), ('nn', self.nn), ('svc', self.svc)], 
            voting='soft' # switching to hard voting bc NN was dominating soft votes with overconfident probabilities
        )
        self.eclf.fit(data, target)
        return self

    def predict(self, data, legal=None):
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # doing this to see the individual probabilities for each model
        results = []
        ensemble_proba = self.eclf.predict_proba(data)
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
                )[0], 2)
            results.append(sample_info)
        
        print(results)
        preds = self.eclf.predict(data)
        return preds
