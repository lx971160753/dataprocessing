import tensorflow as tf

#dir
data_dir="/media/lix/51d7efd5-72f0-450c-ab09-f65259f11e25/dataforc/newlabeldataforc/S*"
save_dir="/home/PycharmProjects/methodc/saved/resnet.ckpt"
#train Parameters
training_iters = 200000000
display_step = 2
learning_rate = 0.001

# Network Parameters
n_input = 3*7*48 # data input
n_classes = 2 # total classes
dropout = 0.5 # Dropout, probability to keep units
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


#batch Parameters
minafterdequeue = 50000
batchsize = 1000
capacity = minafterdequeue + 3*batchsize

# initializing the weights and biases
weights_dic={
    'topconv': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    'wc1_1': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    'wc1_2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    'wc2_1': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    'wc2_2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    'wc3_1': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    'wc3_2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    'wc4_1': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wc4_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'shortcut4':tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wc5_1': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc5_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc6_1': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc6_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc7_1': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc7_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc8_1': tf.Variable(tf.random_normal([3, 3, 64,128])),
    'wc8_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'shortcut8':tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc9_1': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc9_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc10_1': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc10_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc11_1': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc11_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc12_1': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc12_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc13_1': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc13_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc14_1': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc14_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'shortcut14':tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc15_1': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wc15_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wc16_1': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wc16_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'dense1': tf.Variable(tf.random_normal([1, 1, 256, 50])),
    'dense2': tf.Variable(tf.random_normal([6*7*50, 250])),
    'dense3': tf.Variable(tf.random_normal([250, 250])),
    'out':tf.Variable(tf.random_normal([250,2]))
}
biases_dic={
    'topconv': tf.Variable(tf.random_normal([32])),
    'bc1_1': tf.Variable(tf.random_normal([32])),
    'bc1_2': tf.Variable(tf.random_normal([32])),
    'bc2_1': tf.Variable(tf.random_normal([32])),
    'bc2_2': tf.Variable(tf.random_normal([32])),
    'bc3_1': tf.Variable(tf.random_normal([32])),
    'bc3_2': tf.Variable(tf.random_normal([32])),
    'bc4_1': tf.Variable(tf.random_normal([64])),
    'bc4_2': tf.Variable(tf.random_normal([64])),
    'shortcut4':tf.Variable(tf.random_normal([64])),
    'bc5_1': tf.Variable(tf.random_normal([64])),
    'bc5_2': tf.Variable(tf.random_normal([64])),
    'bc6_1': tf.Variable(tf.random_normal([64])),
    'bc6_2': tf.Variable(tf.random_normal([64])),
    'bc7_1': tf.Variable(tf.random_normal([64])),
    'bc7_2': tf.Variable(tf.random_normal([64])),
    'bc8_1': tf.Variable(tf.random_normal([128])),
    'bc8_2': tf.Variable(tf.random_normal([128])),
    'shortcut8':tf.Variable(tf.random_normal([128])),
    'bc9_1': tf.Variable(tf.random_normal([128])),
    'bc9_2': tf.Variable(tf.random_normal([128])),
    'bc10_1': tf.Variable(tf.random_normal([128])),
    'bc10_2': tf.Variable(tf.random_normal([128])),
    'bc11_1': tf.Variable(tf.random_normal([128])),
    'bc11_2': tf.Variable(tf.random_normal([128])),
    'bc12_1': tf.Variable(tf.random_normal([128])),
    'bc12_2': tf.Variable(tf.random_normal([128])),
    'bc13_1': tf.Variable(tf.random_normal([128])),
    'bc13_2': tf.Variable(tf.random_normal([128])),
    'bc14_1': tf.Variable(tf.random_normal([256])),
    'bc14_2': tf.Variable(tf.random_normal([256])),
    'shortcut14':tf.Variable(tf.random_normal([256])),
    'bc15_1': tf.Variable(tf.random_normal([256])),
    'bc15_2': tf.Variable(tf.random_normal([256])),
    'bc16_1': tf.Variable(tf.random_normal([256])),
    'bc16_2': tf.Variable(tf.random_normal([256])),
    'dense1': tf.Variable(tf.random_normal([50])),
    'dense2': tf.Variable(tf.random_normal([250])),
    'dense3': tf.Variable(tf.random_normal([250])),
    'out':tf.Variable(tf.random_normal([2]))
}

