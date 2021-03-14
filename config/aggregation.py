import yaml

cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

def cloud_aggregate(accumulative_gradients):
    return len(accumulative_gradients) < cfg['max_cloud_gradients']

def edge_aggregate(accumulative_gradients):
    return len(accumulative_gradients) < cfg['max_edge_gradients']