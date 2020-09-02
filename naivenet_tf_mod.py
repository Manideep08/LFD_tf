"""
NaiveNet is in a style which is a deliberately very simple
convolutional neural network backbone aiming at deploying 
on all platforms easily, but can also get balance on accuracy 
and efficient extremely at the same time. The entire backbone 
only consists of conv 3×3, conv 1×1, ReLU and residual connection.
"""

# References used to build this
# https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/
# https://stackoverflow.com/questions/52826134/keras-model-subclassing-examples
# https://github.com/keras-team/keras-io/blob/master/tf/custom_layers_and_models.ipynb

import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, ReLU, Add, Layer

num_filters_list = [32, 64, 128, 256]

def conv3x3(num_filters, stride=0, padding='valid'):
    """3x3 convolution with padding"""
    return Conv2D(num_filters, 3, stride, padding, use_bias=True, data_format='channels_first', bias_initializer='glorot_uniform')

def conv1x1(num_filters, padding='valid'):
    """1x1 convolution"""
    return Conv2D(num_filters, 1, 1, padding, use_bias=True, data_format='channels_first', bias_initializer='glorot_uniform')


class BranchNet(Layer):
    """
    The branch of NaiveNet is the network output and 
    only consists of conv 1×1 and ReLU.
    """
    def __init__(self, num_filters):
        super(BranchNet, self).__init__()
        self.conv1 = conv1x1(num_filters)
        self.relu1 = ReLU()
        self.conv2_score = conv1x1(num_filters)
        self.relu2 = ReLU()
        self.conv3_score = conv1x1(2)
        self.conv2_bbox = conv1x1(num_filters)
        self.relu3 = ReLU()
        self.conv3_bbox = conv1x1(4)

    def call(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        out_score = self.conv2_score(out)
        out_score = self.relu2(out_score)
        out_score = self.conv3_score(out_score)

        out_bbox = self.conv2_bbox(out)
        out_bbox = self.relu3(out_bbox)
        out_bbox = self.conv3_bbox(out_bbox)

        return out_score, out_bbox

class NaiveNet(Model):
    """NaiveNet for Fast Single Class Object Detection. 
    The entire backbone and branches only consists of conv 3×3, 
    conv 1×1, ReLU and residual connection.
    """
    def __init__(self, num_filters_list):
        super(NaiveNet, self).__init__()

        self.conv1 = conv3x3(num_filters_list[1], stride=2)
        self.relu1 = ReLU()
        self.conv2 = conv3x3(num_filters_list[1], stride=2)
        self.relu2 = ReLU()
        self.conv3 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu3 = ReLU()
        self.conv4 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu4 = ReLU()
        self.conv5 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu5 = ReLU()
        self.conv6 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu6 = ReLU()
        self.conv7 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu7 = ReLU()
        self.conv8 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu8 = ReLU()
        self.conv9 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu9 = ReLU()
        self.conv10 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu10 = ReLU()

        self.branch_1 = BranchNet(num_filters_list[2])

        self.conv11 = conv3x3(num_filters_list[1], stride=2)
        self.relu11 = ReLU()
        self.conv12 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu12 = ReLU()
        self.conv13 = conv3x3(num_filters_list[1], stride=1, padding='same')
        self.relu13 = ReLU()

        self.branch_2 = BranchNet(num_filters_list[2])
        
        self.conv14 = conv3x3(num_filters_list[2], stride=1, padding='same')
        self.relu14 = ReLU()
        self.conv15 = conv3x3(num_filters_list[2], stride=1, padding='same')
        self.relu15 = ReLU()
        self.conv16 = conv3x3(num_filters_list[2], stride=1, padding='same')
        self.relu16 = ReLU()

        self.branch_3 = BranchNet(num_filters_list[2])

    def call(self, x):
        c1 = self.conv1(x)
        r1 = self.relu1(c1)
        c2 = self.conv2(r1)
        r2 = self.relu2(c2)
        c3 = self.conv3(r2)
        r3 = self.relu3(c3)
        c4 = self.conv4(r3)
        c4 = Add()([c2, c4])
        r4 = self.relu4(c4)
        c5 = self.conv5(r4)
        r5 = self.relu5(c5)
        c6 = self.conv6(r5)
        c6 = Add()([c4, c6])
        r6 = self.relu6(c6)
        c7 = self.conv7(r6)
        r7 = self.relu7(c7)
        c8 = self.conv8(r7)
        c8 = Add()([c6, c8])
        r8 = self.relu8(c8)
        c9 = self.conv9(r8)
        r9 = self.relu9(c9)
        c10 = self.conv10(r9)
        c10 = Add()([c8, c10])
        r10 = self.relu10(c10)

        loss1, box1 = self.branch_1(r10)

        c11 = self.conv11(r10)
        r11 = self.relu11(c11)
        c12 = self.conv12(r11)
        r12 = self.relu12(c12)
        c13 = self.conv13(r12)
        c13 = Add()([c11, c13])
        r13 = self.relu13(c13)

        loss2, box2 = self.branch_2(r13)

        c14 = self.conv14(r13)
        r14 = self.relu14(c14)
        c15 = self.conv15(r14)
        r15 = self.relu15(c15)
        c16 = self.conv16(r15)
        c16 = Add()([c14, c16])
        r16 = self.relu16(c16)

        loss3, box3 = self.branch_3(r16)
        
        return [loss1, box1, loss2, box2, loss3, box3]

def get_model():
    model = NaiveNet(num_filters_list)
    # model.build((32, 3, 320, 320))
    return model

# model = NaiveNet(num_filters_list)
# model.build((32, 3, 320, 320))
# model.summary()