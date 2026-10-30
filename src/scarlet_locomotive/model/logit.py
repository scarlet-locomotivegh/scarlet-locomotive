import numpy as np
import torch
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=100, batch_size=64, seed=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        #list to save the loss after each epoch
        self.loss_list = []

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def cross_entropy(self, y_prob, y):
        #keep the probability away from 0 and 1 so log does not blow up
        y_prob = torch.clamp(y_prob, 1e-7, 1 - 1e-7)
        loss = -(y * torch.log(y_prob) + (1 - y) * torch.log(1 - y_prob))
        return torch.mean(loss)

    def forward(self, x):
        #h(x) = sigmoid(b + w^T x)
        z = self.b + x @ self.w
        return self.sigmoid(z)

    def fit(self, x, y):
        torch.manual_seed(self.seed)

        #turn the data into tensors
        x = torch.tensor(np.array(x), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).flatten()

        #scale the features using the training data
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0)
        x = (x - self.x_mean) / self.x_std

        #weights and bias start at 0
        self.w = torch.zeros(x.shape[1], requires_grad=True)
        self.b = torch.tensor(0.0, requires_grad=True)

        #stochastic gradient descent
        optimizer = torch.optim.SGD([self.w, self.b], lr=self.learning_rate)

        n = x.shape[0]
        for epoch in range(self.epochs):
            #shuffle the data every epoch
            order = torch.randperm(n)

            #go through the data in small batches
            for i in range(0, n, self.batch_size):
                batch = order[i:i + self.batch_size]
                y_prob = self.forward(x[batch])
                loss = self.cross_entropy(y_prob, y[batch])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #loss on all the training data after this epoch
            with torch.no_grad():
                loss = self.cross_entropy(self.forward(x), y)
            self.loss_list.append(loss.item())

        self.final_loss = self.loss_list[-1]
        return self

    def predict_proba(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float32)
        x = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            y_prob = self.forward(x)
        return y_prob.numpy()

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).astype(int)

    def evaluate(self, x, y):
        y = np.array(y).flatten()
        y_prob = self.predict_proba(x)
        y_pred = (y_prob >= 0.5).astype(int)

        #confusion matrix numbers
        tp = np.sum((y_pred == 1) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))

        accuracy = (tp + tn) / len(y)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        specificity = tn / (tn + fp)

        #roc auc = chance a random positive gets a higher probability than a random negative
        order = np.argsort(y_prob)
        rank = np.empty(len(y))
        rank[order] = np.arange(1, len(y) + 1)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        auc = (np.sum(rank[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        print("confusion matrix")
        print("               pred 0   pred 1")
        print("actual 0  ", str(tn).rjust(8), str(fp).rjust(8))
        print("actual 1  ", str(fn).rjust(8), str(tp).rjust(8))
        print()
        print("accuracy    =", round(accuracy, 4))
        print("precision   =", round(precision, 4))
        print("recall      =", round(recall, 4))
        print("specificity =", round(specificity, 4))
        print("f1 score    =", round(f1, 4))
        print("roc auc     =", round(auc, 4))

        return {"accuracy": accuracy, "precision": precision, "recall": recall,
                "specificity": specificity, "f1": f1, "auc": auc}

    def print_weights(self, names=None):
        w = self.w.detach().numpy()
        if names is None:
            names = ["x" + str(i + 1) for i in range(len(w))]
        print("bias =", round(self.b.item(), 4))
        for i in range(len(w)):
            print(names[i], "=", round(w[i], 4))

    def loss_plot(self):
        plt.plot(self.loss_list)
        plt.title("Cross Entropy Loss vs Epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

    def roc_plot(self, x, y):
        y = np.array(y).flatten()
        y_prob = self.predict_proba(x)

        #true positive rate and false positive rate at each threshold
        tpr = []
        fpr = []
        for t in np.linspace(1, 0, 101):
            y_pred = (y_prob >= t).astype(int)
            tpr.append(np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1))
            fpr.append(np.sum((y_pred == 1) & (y == 0)) / np.sum(y == 0))

        plt.plot(fpr, tpr, label="model")
        plt.plot([0, 1], [0, 1], "--", color="gray", label="random guess")
        plt.title("ROC Curve")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend()
        plt.show()
