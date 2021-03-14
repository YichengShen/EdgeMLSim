from mxnet import gluon

MODEL = gluon.nn.Sequential()
with MODEL.name_scope():
    MODEL.add(gluon.nn.Dense(128, in_units=784, activation='relu'))
    MODEL.add(gluon.nn.Dense(64, in_units=128, activation='relu'))
    MODEL.add(gluon.nn.Dense(10, in_units=64))