import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class BottleNeck(nn.Module):
    expansion = 2
    def __init__(self,in_channel,channel,stride=1,downsample=None):
        super().__init__()

        self.conv1=nn.Conv1d(in_channel,channel,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm1d(channel)

        self.conv2=nn.Conv1d(channel,channel,kernel_size=3,padding=1,bias=False,stride=1)
        self.bn2=nn.BatchNorm1d(channel)

        self.conv3=nn.Conv1d(channel,channel*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm1d(channel*self.expansion)

        self.relu=nn.ReLU(False)

        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.relu(self.bn1(self.conv1(x))) #bs,c,h,w
        out=self.relu(self.bn2(self.conv2(out))) #bs,c,h,w
        out=self.relu(self.bn3(self.conv3(out))) #bs,4c,h,w

        if(self.downsample != None):
            residual=self.downsample(residual)

        out = out + residual
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self,block,layers,in_channel):
        super().__init__()
        #定义输入模块的维度
        self.in_channel=in_channel
        ### stem layer
        self.conv1=nn.Conv1d(1,in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm1d(in_channel)
        self.relu=nn.ReLU(False)
        self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2,padding=0,ceil_mode=True)

        ### main layer
        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)

    def forward(self,x):
        ##stem layer
        out=self.relu(self.bn1(self.conv1(x))) #bs,112,112,64
        out=self.maxpool(out) #bs,56,56,64

        ##layers:
        out=self.layer1(out) #bs,56,56,64*4
        out=self.layer2(out) #bs,28,28,128*4
        out=self.layer3(out) #bs,14,14,256*4
        out=self.layer4(out) #bs,7,7,512*4

        return out

    def _make_layer(self,block,channel,blocks,stride=1):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和x的channel维度不匹配问题，即对残差结构的输入进行升维，在做残差相加的时候，必须保证残差的纬度与真正的输出维度（宽、高、以及深度）相同
        # 比如步长！=1 或者 in_channel!=channel&self.expansion
        downsample = None
        if(stride!=1 or self.in_channel!=channel*block.expansion):
            self.downsample=nn.Conv1d(self.in_channel,channel*block.expansion,stride=stride,kernel_size=1,bias=False)
        #第一个conv部分，可能需要downsample
        layers=[]
        layers.append(block(self.in_channel,channel,downsample=self.downsample,stride=stride))
        self.in_channel=channel*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_channel,channel))
        return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000):
        super().__init__()
        #定义输入模块的维度
        self.in_channel=64
        ### stem layer
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(False)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True)

        ### main layer
        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)

        #classifier
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Linear(512*block.expansion,num_classes)
        self.softmax=nn.Softmax(-1)

    def forward(self,x):
        ##stem layer
        out=self.relu(self.bn1(self.conv1(x))) #bs,112,112,64
        out=self.maxpool(out) #bs,56,56,64

        ##layers:
        out=self.layer1(out) #bs,56,56,64*4
        out=self.layer2(out) #bs,28,28,128*4
        out=self.layer3(out) #bs,14,14,256*4
        out=self.layer4(out) #bs,7,7,512*4

        ##classifier
        out=self.avgpool(out) #bs,1,1,512*4
        out=out.reshape(out.shape[0],-1) #bs,512*4
        out=self.classifier(out) #bs,1000
        out=self.softmax(out)

        return out

    def _make_layer(self,block,channel,blocks,stride=1):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和x的channel维度不匹配问题，即对残差结构的输入进行升维，在做残差相加的时候，必须保证残差的纬度与真正的输出维度（宽、高、以及深度）相同
        # 比如步长！=1 或者 in_channel!=channel&self.expansion
        downsample = None
        if(stride!=1 or self.in_channel!=channel*block.expansion):
            self.downsample=nn.Conv2d(self.in_channel,channel*block.expansion,stride=stride,kernel_size=1,bias=False)
        #第一个conv部分，可能需要downsample
        layers=[]
        layers.append(block(self.in_channel,channel,downsample=self.downsample,stride=stride))
        self.in_channel=channel*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_channel,channel))
        return nn.Sequential(*layers)

def ResNet50(num_classes=1000):
    return ResNet(BottleNeck,[3,4,6,3],num_classes=num_classes)


def ResNet101(num_classes=1000):
    return ResNet(BottleNeck,[3,4,23,3],num_classes=num_classes)


def ResNet152(num_classes=1000):
    return ResNet(BottleNeck,[3,8,36,3],num_classes=num_classes)


