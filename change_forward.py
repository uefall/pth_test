# change forward for onnx export
# 需提前在net中定义好
import torch
import torch.nn as nn
import torch.nn.functional as F


class foo_net(nn.Module):
    def __init__(self):
        super(foo_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_new(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


if __name__ == "__main__":
    input = torch.randn(1, 3, 32, 32)
    net = foo_net()
    org_out = net(input)
    torch.onnx.export(net, (input), "org.onnx")
    net.forward = net.forward_new
    new_out = net(input)
    torch.onnx.export(net, (input), "new.onnx")
    print(org_out.shape)
    print(new_out.shape)
