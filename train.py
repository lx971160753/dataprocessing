import tensorflow as tf
import model
import getdata
from parameter import *

data_batch,label_batch=getdata.getTFrecorddata(data_dir,minafterdequeue,batchsize,capacity)
pred= model.ResNet(data_batch, weights_dic, biases_dic)

print('Start training!')
# Define loss and optimizer
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits =pred,labels=label_batch))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(label_batch,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
saver = tf.train.Saver()
init =(tf.global_variables_initializer(),tf.local_variables_initializer())

#start a session
with tf.Session() as sess:
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)


    #summary_writer = tf.train.SummaryWriter('/tmp/logs', graph_def=sess.graph_def)
    step = 1
    # Keep training until reach max iterations
    while step * batchsize < training_iters:
        # Fit training using batch data
        sess.run(optimizer, feed_dict={keep_prob: dropout})
        if step % display_step == 0:
            #print(sess.run(pred))
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={keep_prob: 1.})
            print(
            "Iter " + str(step * batchsize) + ", Minibatch Loss= " + "{:.6f}".format(
                loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    saver.save(sess,save_dir)
    print("Optimization Finished!")