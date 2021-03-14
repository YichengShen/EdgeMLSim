from mxnet import gluon

LOSS = gluon.loss.SoftmaxCrossEntropyLoss()

# LOSS = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# Refer to
# https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/packages/gluon/loss/loss.html
# for more Loss functions