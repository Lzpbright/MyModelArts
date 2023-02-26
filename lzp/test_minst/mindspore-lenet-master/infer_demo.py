
import mindspore.nn as nn
from mindspore.train import Model

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet
from mindvision.engine.callback import LossMonitor


epochs = 10
batch_size = 32
momentum = 0.9
learning_rate = 1e-2


# 1. 构建数据集
download_train = Mnist(path="./mnist", split="train", batch_size=batch_size, repeat_num=1, shuffle=True, resize=32, download=True)
dataset_train = download_train.run()

# 2. 定义神经网络
network = lenet(num_classes=10, pretrained=False)
# 3.1 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 3.2 定义优化器函数
net_opt = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=momentum)
# 3.3 初始化模型参数
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'acc'})

# 4. 对神经网络执行训练
model.train(epochs, dataset_train, callbacks=[LossMonitor(learning_rate, 1875)])
