import json

import mxnet as mx

from collections import OrderedDict


def string_to_tuple(string):
    return string.strip('(').strip(')').split(',')


def combine_feature_maps(intermediate_symbols, node, nodes, scaled_feature):
    data_input_node = nodes[node['inputs'][0][0]]
    try:
        data_input_symbol = intermediate_symbols["{}_output".format(data_input_node['name'])]
        # print("data_input_node =", data_input_node['name'])
    except ValueError:
        data_input_symbol = intermediate_symbols[data_input_node['name']]
        # print("data_input_node =", data_input_node['name'])
    averaged_feature_map = mx.symbol.mean(data_input_symbol, axis=1, keepdims=True)
    # print("averaged_feature_map = {}".format(get_shape(averaged_feature_map)))
    # print("scaled_feature = {}".format(get_shape(scaled_feature)))
    feature_map = scaled_feature * averaged_feature_map
    # print("feature_map = {}".format(get_shape(feature_map)))
    return feature_map


def get_all_layer_names(nodes):
    layers = OrderedDict()
    for i, node in enumerate(nodes):
        if node['op'] != 'null':
            layers[node['name']] = i
    return layers


def reduce_symbol_json(json_str, layer_name):
    symbol_json = json.loads(json_str)
    layers = get_all_layer_names(symbol_json['nodes'])

    # strip all unecessary parts of network and add new loss layers
    head_id = layers[layer_name]
    symbol_json['nodes'] = symbol_json['nodes'][:head_id + 1]
    symbol_json['arg_nodes'] = list(filter(lambda x: x <= head_id, symbol_json['arg_nodes']))
    symbol_json['heads'] = [[len(symbol_json['nodes']) - 1, 0]]

    return json.dumps(symbol_json)


def is_conv_or_pool(node):
    return 'Convolution' in node['op'] or node['op'] == 'Pooling'


def prev_conv_or_pool_node(node_name, reversed_nodes, input_name):
    prev_node = None

    for i, node in enumerate(reversed_nodes):
        if input_name is not None and node['name'] == input_name or node['name'] == 'data':
            break

        if not is_conv_or_pool(node):
            continue

        if node_name == node['name']:
            return prev_node
        prev_node = node

    return None


def get_shape(symbol, data_shape=(1, 3, 224, 224)):
    shapes = mx.symbol_doc.SymbolDoc.get_output_shape(symbol, data=data_shape)
    return next(iter(shapes.values()))


def build_visual_backprop_symbol(start_symbol, input_name=None, data_shape=(1, 3, 224, 224)):
    complete_json = start_symbol.tojson()

    computational_graph = json.loads(start_symbol.tojson())
    nodes = computational_graph['nodes']

    assert nodes[-1]['op'] == 'Activation', 'Visual Backprop needs an activation node as starting point!'

    intermediate_symbols = start_symbol.get_internals()

    feature_map = mx.symbol.mean(start_symbol, axis=1, keepdims=True)

    nodes_to_process = []
    for node in reversed(nodes):
        if input_name is not None and node['name'] == input_name or node['name'] == 'data':
            break
        if node['name'].endswith("_sc"):
            continue
        if not is_conv_or_pool(node):
            continue
        nodes_to_process.append(node)
    # print(list(x['name'] for x in nodes_to_process))

    for i, node in enumerate(nodes_to_process):
        # print("=" * 40)
        node_attrs = node.get('attrs', None)

        # if i + 2 >= len(nodes_to_process):
        #     target_shape = data_shape[-2:]
        # else:
        #     node_symbol = mx.symbol.load_json(reduce_symbol_json(complete_json, nodes_to_process[i + 2]['name']))
        #     target_shape = get_shape(node_symbol, data_shape)[-2:]
        # print("target_shape: {}".format(target_shape))

        # feature_map_shape = get_shape(feature_map, data_shape)[-2:]
        # print("feature_map_shape: {}".format(feature_map_shape))

        adjustment = (0, 0)
        kernel_height, kernel_width = map(int, string_to_tuple(node_attrs['kernel']))
        stride_height, stride_width = map(int, string_to_tuple(node_attrs['stride']))
        pad_height, pad_width = map(int, string_to_tuple(node_attrs.get('pad', '(0, 0)')))
        if stride_height == 2 and not kernel_height == 2:
            adjustment = (1, 1)

        # print("name =", node['name'])
        # print("op =", node['op'])
        # print("kernel =", kernel_height, kernel_width)
        # print("stride =", stride_height, stride_width)
        # print("pad =", pad_height, pad_width)

        scaled_feature = mx.symbol.Deconvolution(
            data=feature_map,
            weight=mx.symbol.ones((1, 1, kernel_height, kernel_width)),
            kernel=(kernel_height, kernel_width),
            stride=(stride_height, stride_width),
            pad=(pad_height, pad_width),
            num_filter=1,
            adj=adjustment,
        )
        output_shape = get_shape(scaled_feature, data_shape)[-2:]
        # print("scaled_feature: {}".format(output_shape))

        feature_map = combine_feature_maps(intermediate_symbols, node, nodes, scaled_feature)

    # normalize feature map
    min_value = mx.symbol.min(feature_map)
    max_value = mx.symbol.max(feature_map)
    feature_map = mx.symbol.broadcast_sub(feature_map, min_value)
    feature_map = mx.symbol.broadcast_mul(feature_map, 1.0 / (max_value - min_value))
    # print(feature_map.tojson())
    return feature_map
