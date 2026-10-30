import numpy as np
import pandas as pd
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


class CauchyRegression:

    def __init__(self, c=1.0, learning_rate=0.1, epochs=2000):
        self.c = c
        self.learning_rate = learning_rate
        self.epochs = epochs

        #list to save the loss while training
        self.loss_list = []

    def cauchy_loss(self, y_pred, y):
        #cauchy loss = c^2 / 2 * log(1 + ((y - y_pred) / c)^2)
        r = (y - y_pred) / self.c
        loss = (self.c ** 2 / 2) * torch.log(1 + r ** 2)
        return torch.mean(loss)

    def forward(self, x):
        #y = w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4
        y = self.w0 + x @ self.w
        return y

    def fit(self, x, y):
        #turn the data into tensors
        x = torch.tensor(np.array(x), dtype=torch.float64)
        y = torch.tensor(np.array(y), dtype=torch.float64).flatten()

        #scale the features so gradient descent works
        #(pressure is around 1000 and temperature is around 20)
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0)
        x = (x - self.x_mean) / self.x_std

        #weights start at 0 and w0 starts at the median of y
        self.w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        self.w0 = torch.tensor(y.median().item(), dtype=torch.float64, requires_grad=True)

        #Adam optimizer
        optimizer = torch.optim.Adam([self.w0, self.w], lr=self.learning_rate)

        for epoch in range(self.epochs):
            #predict and find the loss
            y_pred = self.forward(x)
            loss = self.cauchy_loss(y_pred, y)

            #gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.loss_list.append(loss.item())

        self.final_loss = self.loss_list[-1]

        #change the weights back to the original units
        with torch.no_grad():
            coef = self.w / self.x_std
            intercept = self.w0 - torch.sum(self.w * self.x_mean / self.x_std)
        self.coefficients = torch.cat([intercept.reshape(1), coef]).numpy()

        return self.coefficients

    def predict(self, x):
        x = torch.tensor(np.array(x), dtype=torch.float64)
        x = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            y = self.forward(x)
        return y.numpy()

    def print_coefficients(self, names=None):
        if names is None:
            names = ["x" + str(i + 1) for i in range(len(self.coefficients) - 1)]
        print("w0 (intercept) =", round(self.coefficients[0], 4))
        for i in range(len(names)):
            print("w" + str(i + 1), "(" + names[i] + ") =", round(self.coefficients[i + 1], 4))
        print("final loss =", round(self.final_loss, 4))

    def confidence_interval(self, x, y):
        #add a column of 1s for w0
        x = torch.tensor(np.array(x), dtype=torch.float64)
        y = torch.tensor(np.array(y), dtype=torch.float64).flatten()
        x = torch.cat([torch.ones(x.shape[0], 1, dtype=torch.float64), x], dim=1)
        params = torch.tensor(self.coefficients, dtype=torch.float64)

        #negative log likelihood if the errors follow a cauchy distribution with scale c
        def nll(p):
            r = (y - x @ p) / self.c
            return torch.sum(torch.log(1 + r ** 2))

        #covariance = inverse of the hessian
        H = torch.autograd.functional.hessian(nll, params)
        cov = torch.linalg.inv(H)
        se = torch.sqrt(torch.diag(cov)).numpy()

        lower = self.coefficients - 1.96 * se
        upper = self.coefficients + 1.96 * se
        self.std_error = se
        self.ci = np.column_stack([lower, upper])
        return se, lower, upper

    def scatter_plot(self, df):
        #scatterplot matrix of every column against every other column
        pd.plotting.scatter_matrix(df, figsize=(12, 12), alpha=0.2, diagonal="hist")
        plt.suptitle("Scatterplot Matrix")
        plt.show()

        #correlation numbers
        print(df.corr().round(3))

    def loss_plot(self):
        plt.plot(self.loss_list)
        plt.title("Cauchy Loss vs Epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

    def residual_plot(self, x, y, names=None):
        x = np.array(x)
        y = np.array(y).flatten()
        residual = self.predict(x) - y

        if names is None:
            names = ["x" + str(i + 1) for i in range(x.shape[1])]

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax = ax.flatten()
        for i in range(x.shape[1]):
            ax[i].scatter(x[:, i], residual, alpha=0.2, s=5)
            ax[i].axhline(0, color="red")
            ax[i].set_title("Residual vs " + names[i])
            ax[i].set_xlabel(names[i])
            ax[i].set_ylabel("y_hat - y")
        plt.tight_layout()
        plt.show()
