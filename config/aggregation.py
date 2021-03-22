import yaml

cfg = yaml.load(open('config/config.yml', 'r'), Loader=yaml.FullLoader)

def cloud_aggregation_condition(accumulative_gradients):
    return len(accumulative_gradients) >= cfg['max_cloud_gradients']

def edge_aggregation_condition(accumulative_gradients):
    return len(accumulative_gradients) >= cfg['max_edge_gradients']