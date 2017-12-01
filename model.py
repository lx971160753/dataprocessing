import tensorflow as tf
import padding
from parameter import weights_dic,biases_dic

def conv2d(name, l_input, w, b):
    return padding.jijipadding(
        tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='VALID'), b), name=name))


def conv2d_(name, l_input, w, b):
    return padding.jijipadding(
        tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 2, 1], padding='VALID'), b), name=name))


def conv2d_2(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 2, 1], padding='SAME'), b), name=name)


def conv2d_1(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


def identity_block(input_tensor, block, weights, biases):
    """
    The identity block is the block that has no conv layer at shortcut.
    """
    conv1 = conv2d('conv' + block + '_1', input_tensor, weights_dic['wc' + block + '_1'], biases_dic['bc' + block + '_1'])
    conv2 = conv2d('conv' + block + '_2', conv1, weights_dic['wc' + block + '_2'], biases_dic['bc' + block + '_2'])
    addshortcut = input_tensor + conv2
    i_output = tf.nn.sigmoid(addshortcut)
    return i_output


def conv_block(input_tensor, block, weights, biases):
    """
    conv_block is the block that has a conv layer at shortcut
    """
    conv1 = conv2d_2('conv' + block + '_1', input_tensor, weights_dic['wc' + block + '_1'], biases_dic['bc' + block + '_1'])
    conv2 = conv2d('conv' + block + '_2', conv1, weights_dic['wc' + block + '_2'], biases_dic['bc' + block + '_2'])
    shortcut = conv2d_2('shortcut' + block, input_tensor, weights_dic['shortcut' + block], biases_dic['shortcut' + block])
    addshortcut = shortcut + conv2
    c_output = tf.nn.sigmoid(addshortcut)
    return c_output


def ResNet(inputs, _weights, _biases):
    x = tf.transpose(tf.reshape(inputs, shape=[-1, 3, 7, 48]), perm=[0, 2, 3, 1])

    x1 = conv2d('topconv', x, w=_weights['topconv'], b=_biases['topconv'])

    x2 = identity_block(input_tensor=x1, block='1', weights=_weights, biases=_biases)
    x3 = identity_block(input_tensor=x2, block='2', weights=_weights, biases=_biases)
    x4 = identity_block(input_tensor=x3, block='3', weights=_weights, biases=_biases)

    x5 = conv_block(input_tensor=x4, block='4', weights=_weights, biases=_biases)
    x6 = identity_block(input_tensor=x5, block='5', weights=_weights, biases=_biases)
    x7 = identity_block(input_tensor=x6, block='6', weights=_weights, biases=_biases)
    x8 = identity_block(input_tensor=x7, block='7', weights=_weights, biases=_biases)

    x9 = conv_block(input_tensor=x8, block='8', weights=_weights, biases=_biases)
    x10 = identity_block(input_tensor=x9, block='9', weights=_weights, biases=_biases)
    x11 = identity_block(input_tensor=x10, block='10', weights=_weights, biases=_biases)
    x12 = identity_block(input_tensor=x11, block='11', weights=_weights, biases=_biases)
    x13 = identity_block(input_tensor=x12, block='12', weights=_weights, biases=_biases)
    x14 = identity_block(input_tensor=x13, block='13', weights=_weights, biases=_biases)

    x15 = conv_block(input_tensor=x14, block='14', weights=_weights, biases=_biases)
    x16 = identity_block(input_tensor=x15, block='15', weights=_weights, biases=_biases)
    x17 = identity_block(input_tensor=x16, block='16', weights=_weights, biases=_biases)

    # transporting the tensor's format from (6,7,256) to (6,7,50),so that the softmax can have less parameters
    x18 = conv2d_1('dense1', x17, _weights['dense1'], _biases['dense1'])

    # To flatten the tensor to 'list'
    x19 = tf.reshape(x18, [-1, _weights['dense2'].get_shape().as_list()[0]])
    x20 = tf.nn.relu(tf.matmul(x19, _weights['dense2']) + _biases['dense2'], name='dense2')
    x21 = tf.nn.relu(tf.matmul(x20, _weights['dense3']) + _biases['dense3'], name='dense3')
    out = tf.matmul(x21, _weights['out']) + _biases['out']
    return out


