import tensorflow as tf
import tensorflow_hub as hub

module_link = {'inception_v3': 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'}

def load_module(module_name = 'inception_v3'):
    # Tested only for inception_V3
    if module_name == 'inception_v3':
        module = hub.Module(module_link[module_name])
        module_input = module.get_input_info_dict()
        module_input_shape = module_input['images'].get_shape()
        image_shape = (int(module_input_shape[1]), int(module_input_shape[2]), int(module_input_shape[3]))
        input_tensor = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]])
        output_tensor = module(input_tensor)
        return (input_tensor, output_tensor, image_shape)