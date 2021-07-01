import yaml
from mxnet import gluon, nd
import byzantine

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
#             Aggregation Condition            #
################################################
def cloud_aggregation_condition(accumulative_gradients):
    return len(accumulative_gradients) >= cfg['max_cloud_gradients']


def edge_aggregation_condition(accumulative_gradients):
    return len(accumulative_gradients) >= cfg['max_edge_gradients']


################################################
#              Aggregation Method              #
################################################

AGGREGATION_METHOD = cfg['aggregation_method']

def aggre(gradients_to_aggregate, byz=byzantine.no_byz):
    # Flatten the gradients
    # param_list shape: (flattened size, n) if there are n gradients
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients_to_aggregate]
    byz(param_list, F, grad_example=gradients_to_aggregate[0]) # See Byzantine section below
    aggregated_gradients = None
    if AGGREGATION_METHOD == "mean":
        aggregated_gradients = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    elif AGGREGATION_METHOD == "marginal_median":
        # Sort param_list
        sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
        # Odd number of columns
        if sorted_array.shape[-1] % 2 == 1:
            mid_idx = int(sorted_array.shape[-1]/2)
            aggregated_gradients = sorted_array[:, mid_idx]
        # Even number of columns
        else:
            mid_idx1 = int(sorted_array.shape[-1]/2-1)
            mid_idx2 = int(sorted_array.shape[-1]/2)
            aggregated_gradients = (sorted_array[:, mid_idx1] + sorted_array[:, mid_idx2]) / 2.
    else:
        print("Undefined aggregation method")
    return aggregated_gradients


################################################
#                  Byzantine                   #
################################################ 
# Byzantine types: no_byz, gaussian_attack, bitflip_attack, signflip_attack (See byzantine.py)   
# 1. At Cloud Server level
if cfg['byzantine_type_cloud'] == 'gaussian_attack':
    BYZ_TYPE_CLOUD = byzantine.gaussian_attack
elif cfg['byzantine_type_cloud'] == 'bitflip_attack':
    BYZ_TYPE_CLOUD = byzantine.bitflip_attack
elif cfg['byzantine_type_cloud'] == 'signflip_attack':
    BYZ_TYPE_CLOUD = byzantine.signflip_attack
else:
    BYZ_TYPE_CLOUD = byzantine.no_byz

# 2. At Edge Server level
if cfg['byzantine_type_edge'] == 'gaussian_attack':
    BYZ_TYPE_EDGE = byzantine.gaussian_attack
elif cfg['byzantine_type_edge'] == 'bitflip_attack':
    BYZ_TYPE_EDGE = byzantine.bitflip_attack
elif cfg['byzantine_type_edge'] == 'signflip_attack':
    BYZ_TYPE_EDGE = byzantine.signflip_attack
else:
    BYZ_TYPE_EDGE = byzantine.no_byz

# Number of Faulty Gradients
F = cfg['num_faulty_grad']


################################################
#                    Loss                      #
################################################
LOSS = gluon.loss.SoftmaxCrossEntropyLoss()

# LOSS = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# Refer to
# https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/packages/gluon/loss/loss.html
# for more Loss functions
