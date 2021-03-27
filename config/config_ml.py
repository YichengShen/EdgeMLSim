import yaml
from mxnet import gluon

cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

################################################
#                     MODEL                    #
################################################
MODEL = gluon.nn.Sequential()
with MODEL.name_scope():
    MODEL.add(gluon.nn.Dense(128, in_units=784, activation='relu'))
    MODEL.add(gluon.nn.Dense(64, in_units=128, activation='relu'))
    MODEL.add(gluon.nn.Dense(10, in_units=64))


################################################
#                  Aggregation                 #
################################################
def cloud_aggregation_condition(accumulative_gradients):
    return len(accumulative_gradients) >= cfg['max_cloud_gradients']


def edge_aggregation_condition(accumulative_gradients):
    return len(accumulative_gradients) >= cfg['max_edge_gradients']


################################################
#                    Loss                      #
################################################
LOSS = gluon.loss.SoftmaxCrossEntropyLoss()

# LOSS = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# Refer to
# https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/packages/gluon/loss/loss.html
# for more Loss functions
