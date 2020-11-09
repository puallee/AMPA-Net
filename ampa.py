import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tflearn.layers.conv import global_max_pool
def Sigmoid(x):
    return tf.nn.sigmoid(x)
def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)
def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape,initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    # Weights = tf.get_variable(shape=w_shape, initializer=tf.keras.initializers.he_normal(),
    #                           name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        network = Relu(network)
        return network

def Relu(x):
    return tf.nn.relu(x)
def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')
def Global_Max_Pooling(x):
    return global_max_pool(x, name='Global_avg_pooling')


def channelattentionnetwork( input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            filter_size=3
            squeeze = Global_Average_Pooling(input_x)
            squeeze1=Global_Max_Pooling(input_x)
            squeeze1=tf.reshape(squeeze1,[-1,1,1,32])
            squeeze=tf.reshape(squeeze,[-1,1,1,32])
            [Weights8, bias8] = add_con2d_weight_bias([1, 1, out_dim, out_dim/ratio], [int(out_dim/ratio)], 8)
            [Weights9, bias9] = add_con2d_weight_bias([1, 1, out_dim/ratio, out_dim], [int(out_dim)], 9)
            excitation=tf.nn.conv2d(squeeze, Weights8, strides=[1, 1, 1, 1], padding='SAME')
            excitation = tf.nn.relu(tf.layers.batch_normalization(excitation))
            excitation=tf.nn.conv2d(excitation,Weights9,strides=[1, 1, 1, 1], padding='SAME')
            excitat = tf.nn.conv2d(squeeze1, Weights8, strides=[1, 1, 1, 1], padding='SAME')
            excitat = tf.nn.relu(tf.layers.batch_normalization(excitat))
            excitat = tf.nn.conv2d(excitat, Weights9, strides=[1, 1, 1, 1], padding='SAME')
            excitation=excitation+excitat
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])

            scale = input_x * excitation

            return scale


def spatialattentionnetwork(input_x, layer_name):
    with tf.name_scope(layer_name):
        squeeze = tf.reduce_mean(input_x, axis=[3])
        squeeze1=tf.reduce_max(input_x, axis=[3])
        squeeze=tf.reshape(squeeze,[-1,33,33,1])
        squeeze1 = tf.reshape(squeeze1, [-1, 33, 33, 1])
        x=tf.concat([squeeze,squeeze1],3)
        excitation = tf.layers.conv2d(inputs=x, use_bias=True, filters=1, kernel_size=[3, 3],padding='SAME',
                                      strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        excitation = Sigmoid(excitation)
        scale = input_x * excitation
        return scale

def ampanet_block(CSRecon,z, PhiT):
    conv_size = 32
    filter_size = 3

    x1_ampa=tf.matmul(z,PhiT)+CSRecon[-1]

    x2_ampa = tf.reshape(x1_ampa, shape=[-1, 33, 33, 1])

    [Weights0, bias0] = add_con2d_weight_bias([3, 3, 1, conv_size], [conv_size], 0)
    [Weights1, bias1] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 1)
    [Weights11, bias11] = add_con2d_weight_bias([filter_size, filter_size, 32, conv_size], [conv_size], 11)
    [Weights2, bias2] = add_con2d_weight_bias([3, 3, conv_size, conv_size], [conv_size], 2)
    [Weights22, bias22] = add_con2d_weight_bias([filter_size, filter_size, 32, conv_size], [conv_size], 22)

    [Weights3, bias3] = add_con2d_weight_bias([3, 3, conv_size, 1], [1], 3)

    x3_ampa = tf.nn.conv2d(x2_ampa, Weights0, strides=[1, 1, 1, 1], padding='SAME')

    x4_ampa = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(x3_ampa, Weights1, strides=[1, 1, 1, 1], padding='SAME')))

    x44_ampa = tf.nn.conv2d(x4_ampa, Weights11, strides=[1, 1, 1, 1], padding='SAME')
    x44_ampa=tf.layers.batch_normalization(x44_ampa)
    x5_ampa=tf.nn.relu(x44_ampa)

    x6_ampa = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(x5_ampa, Weights2, strides=[1, 1, 1, 1], padding='SAME')))

    x66_ampa = tf.nn.conv2d(x6_ampa, Weights22, strides=[1, 1, 1, 1], padding='SAME')
    x66_ampa=spatialattentionnetwork(x66_ampa,layer_name='SA')
    x66_ampa=channelattentionnetwork( x66_ampa, 32, 4, layer_name='SE')
    x7_ampa = tf.nn.conv2d(x66_ampa, Weights3, strides=[1, 1, 1, 1], padding='SAME')
    x7_ampa = x7_ampa + x2_ampa

    x8_ampa = tf.reshape(x7_ampa, shape=[-1, 1089])
    
    x3_ampa_ortho = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(x3_ampa, Weights1, strides=[1, 1, 1, 1], padding='SAME')))
    x4_ampa_ortho = tf.nn.conv2d(x3_ampa_ortho, Weights11, strides=[1, 1, 1, 1], padding='SAME')
    x6_ampa_ortho = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(x4_ampa_ortho, Weights2, strides=[1, 1, 1, 1], padding='SAME')))
    x7_ampa_ortho = tf.nn.conv2d(x6_ampa_ortho, Weights22, strides=[1, 1, 1, 1], padding='SAME')
    x11_ampa_ortho = x7_ampa_ortho - x3_ampa

 
    return [x8_ampa, x11_ampa_ortho]


def inference_ampa(input_tensor,n,PhiT,Phi,y,reuse):
    CSRecon = []
    CSOrtho = []
    CSRecon.append(input_tensor)
    z = y - tf.matmul(input_tensor,Phi)
    for i in range(n):
        with tf.variable_scope('conv_%d' %i, reuse=reuse):
            var_step = tf.Variable(0.1, dtype=tf.float32)
            [x8_ampa, x11_ampa_ortho] = ampanet_block(CSRecon,z,PhiT)
            CSRecon.append(x8_ampa)
            CSOrtho.append(x11_ampa_ortho)
            z=y-tf.matmul(CSRecon[-1],Phi)+tf.scalar_mul(var_step,z)
    return [CSRecon, CSOrtho]
