import tensorflow as tf

"""输入是某一层网络的输出，经过自定义的padding方法后得到输出"""
def jijipadding(x):
    prex=x[:,1:,0,:]
    prex1=x[:,0,0,:]

    posx=x[:,0:-1,-1,:]
    posx1=x[:,-1,-1,:]

    prex=tf.concat([prex,tf.expand_dims(prex1,1)],1)
    posx=tf.concat([tf.expand_dims(posx1,1),posx],1)

    x1=tf.concat([tf.expand_dims(posx,2),x,tf.expand_dims(prex,2)],2)

    topx=x1[:,0,:,:]
    botx=x1[:,-1,:,:]

    x=tf.concat([tf.expand_dims(botx,1),x1,tf.expand_dims(topx,1)],1)

    return x