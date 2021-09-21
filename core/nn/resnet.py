import mxnet as mx

class ResNet:

    @staticmethod
    def residual_module(data, filters, stride, reduce=False, bn_eps=2e-5, bn_momentum=0.9):

        shortcut = data

        # First block
        bn_1_1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act_1_1 = mx.sym.Activation(data=bn_1_1, act_type="relu")
        conv_1_1 = mx.sym.Convolution(data=act_1_1, num_filter=int(filters * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)

        bn_2_1 = mx.sym.BatchNorm(data=conv_1_1, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act_2_1 = mx.sym.Activation(data=bn_2_1, act_type="relu")
        conv_2_1 = mx.sym.Convolution(data=act_2_1, num_filter=int(filters * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True)

        bn_3_1 = mx.sym.BatchNorm(data=conv_2_1, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act_3_1 = mx.sym.Activation(data=bn_3_1, act_type="relu")
        conv_3_1 = mx.sym.Convolution(data=act_3_1, num_filter=filters, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True)

        if reduce:
            shortcut = mx.sym.Convolution(data=act_1_1, num_filter=filters, kernel=(1, 1), stride=stride, pad=(0, 0), no_bias=True)

        output = conv_3_1 + shortcut
        return output



    @staticmethod
    def build(num_classes, stages, filters, bn_eps=2e-5, bn_momentum=0.9):

        data = mx.sym.Variable("data")

        bn_1_1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=bn_eps, momentum=bn_momentum)
        conv_1_1 = mx.sym.Convolution(data=bn_1_1, num_filter=filters[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True)
        bn_1_2 = mx.sym.BatchNorm(data=conv_1_1, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act_1_2 = mx.sym.Activation(data=bn_1_2, act_type="relu")
        maxpool_1_1 = mx.sym.Pooling(data=act_1_2, pool_type="max", kernel=(3, 3), stride=(2, 2), pad=(1, 1))

        output = maxpool_1_1

        for i in range(len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            output = ResNet.residual_module(output, filters[i+1], stride=stride, reduce=True, bn_eps=bn_eps, bn_momentum=bn_momentum)
            for j in range(stages[i] - 1):
                output = ResNet.residual_module(output, filters[i+1], (1, 1), bn_eps=bn_eps, bn_momentum=bn_momentum)

        bn_2_1 = mx.sym.BatchNorm(data=output, fix_gamma=False, eps=bn_eps, momentum=bn_momentum)
        act_2_1 = mx.sym.Activation(data=bn_2_1, act_type="relu")
        avgpool_1_1 = mx.sym.Pooling(data=act_2_1, pool_type="avg", global_pool=True, kernel=(7, 7))
        flatten = mx.sym.Flatten(data=avgpool_1_1)
        fc_2_1 = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes)
        model = mx.sym.SoftmaxOutput(data=fc_2_1, name="softmax")

        return model
