import torch
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd
        return output, None

class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        #input size = batch size x 2048 x 1 x 1


        self.linear_1 = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1,1)),torch.nn.Flatten(),torch.nn.Linear(128, 32), torch.nn.ReLU(True))
        self.linear_2 = torch.nn.Sequential(torch.nn.Linear(32, 1), torch.nn.Sigmoid())


    def forward(self, x, lamda=1):
        
        x = self.grad_reverse(x, lamda)
        x = self.linear_1(x)
        x = self.linear_2(x)

        return x

    def grad_reverse(self, x, lambd=1.0):
        return GradReverse.apply(x, lambd)

class Discriminator_conv(torch.nn.Module):

    def __init__(self):
        super(Discriminator_conv, self).__init__()

        self.conv1 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=10, stride=2), torch.nn.ReLU(True))
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=10, stride=2), torch.nn.ReLU(True))
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.conv3 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=10, stride=2), torch.nn.ReLU(True))
        self.bn3 = torch.nn.BatchNorm1d(16)
        self.conv4 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=16, out_channels=8, kernel_size=10, stride=2), torch.nn.ReLU(True))
        self.bn4 = torch.nn.BatchNorm1d(8)
        self.conv5 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=10, stride=2), torch.nn.ReLU(True))
        self.bn5 = torch.nn.BatchNorm1d(4)
        self.conv6 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=4, out_channels=2, kernel_size=10, stride=2), torch.nn.ReLU(True))
        self.bn6 = torch.nn.BatchNorm1d(2)
        self.linear1 = torch.nn.Sequential(torch.nn.Linear(240, 128), torch.nn.ReLU(True))
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(128, 16), torch.nn.ReLU(True))
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(16, 1), torch.nn.Sigmoid())


    def forward(self, x, lamda=1):
        
        x = self.grad_reverse(x, lamda)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x

    def grad_reverse(self, x, lambd=1.0):
        return GradReverse.apply(x, lambd)

if __name__ == '__main__':
    discriminator_model = Discriminator()
   
    input = torch.randn(8, 128, 64, 128)
    # input = torch.nn.AdaptiveAvgPool2d((1, 1))(input)
    # print(input.shape)
    output = discriminator_model(input)

    print(output.shape)
    