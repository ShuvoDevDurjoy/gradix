from gradix.nn.modules import Conv2D, Linear, Module


class MNISTModel(Module):

    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def forward(self, x):
        x = x.reshape(x.shape[0], 784)
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x



class MNISTConvModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(1, 8, kernel=3, stride=2, padding=1)
        self.conv2 = Conv2D(8, 16, kernel=3, stride=2, padding=1)
        self.conv3 = Conv2D(16, 16, kernel=3, padding=1)
        self.fc = Linear(16 * 7 * 7, 10)

    def parameters(self):
        return (
            self.conv1.parameters()
            + self.conv2.parameters()
            + self.conv3.parameters()
            + self.fc.parameters()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = x.reshape(x.shape[0], 16 * 7 * 7)
        return self.fc(x)