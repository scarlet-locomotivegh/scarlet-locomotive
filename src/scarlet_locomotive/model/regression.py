import numpy as np
import torch
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        #w0 and w1 start at 0
        self.w0 = torch.tensor(0.0, requires_grad=True)
        self.w1 = torch.tensor(0.0, requires_grad=True)

        #SGD optimizer and mean squared error loss
        self.optimizer = torch.optim.SGD([self.w0, self.w1], lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()

        #lists to save w0, w1, and loss while training
        self.w0_list = []
        self.w1_list = []
        self.loss_list = []

    def forward(self, x):
        #linear model y = w0 + w1 * x
        y = self.w0 + self.w1 * x
        return y

    def fit(self, x_train, y_train, x_test, y_test):
        #turn the data into tensors
        x_train = torch.tensor(np.array(x_train), dtype=torch.float32).flatten()
        y_train = torch.tensor(np.array(y_train), dtype=torch.float32).flatten()
        x_test = torch.tensor(np.array(x_test), dtype=torch.float32).flatten()
        y_test = torch.tensor(np.array(y_test), dtype=torch.float32).flatten()

        for epoch in range(self.epochs):
            #predict and find the loss
            y_pred = self.forward(x_train)
            loss = self.loss_function(y_pred, y_train)

            #gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #save the values
            self.w0_list.append(self.w0.item())
            self.w1_list.append(self.w1.item())
            self.loss_list.append(loss.item())

        #R2 on the test data
        with torch.no_grad():
            y_pred = self.forward(x_test)
            ss_res = torch.sum((y_test - y_pred) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot

        print("w0 =", self.w0.item())
        print("w1 =", self.w1.item())
        print("R2 on test data =", r2.item())
        return r2.item()

    def predict(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float32).flatten()
        with torch.no_grad():
            y = self.forward(x)
        return y.numpy()

    def analysis_plot(self, x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # original data and the fitted line
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = self.predict(line_x)
        ax[0, 0].scatter(x, y, label="data")
        ax[0, 0].plot(line_x, line_y, color="red", label="fitted line")
        ax[0, 0].set_title("Data and Fitted Line")
        ax[0, 0].set_xlabel("x")
        ax[0, 0].set_ylabel("y")
        ax[0, 0].legend()

        # w0 while training
        ax[0, 1].plot(self.w0_list)
        ax[0, 1].set_title("w0 vs Epoch")
        ax[0, 1].set_xlabel("epoch")
        ax[0, 1].set_ylabel("w0")

        # w1 while training
        ax[1, 0].plot(self.w1_list)
        ax[1, 0].set_title("w1 vs Epoch")
        ax[1, 0].set_xlabel("epoch")
        ax[1, 0].set_ylabel("w1")

        # loss while training
        ax[1, 1].plot(self.loss_list)
        ax[1, 1].set_title("Loss vs Epoch")
        ax[1, 1].set_xlabel("epoch")
        ax[1, 1].set_ylabel("MSE loss")

        plt.tight_layout()
        plt.show()
