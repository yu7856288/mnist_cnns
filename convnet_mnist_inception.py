import os
from keras.utils.conv_utils import convert_kernel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time 

import tensorflow as tf

import utils
"""
实现一个inception结构的cnn
"""

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    A method that does convolution + relu on inputs
    '''
    with tf.variable_scope(name_or_scope=scope_name, reuse=tf.AUTO_REUSE)  as scope:
        input_channels = inputs.shape[-1]  # ##取shape的最后一维
        kernel = tf.get_variable(name='kernel', shape=[k_size, k_size, input_channels, filters], initializer=tf.truncated_normal_initializer)
        biases = tf.get_variable(name='biases', shape=[filters], initializer=tf.zeros_initializer)
        conv1 = tf.nn.conv2d(input=inputs, filter=kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv1) + biases


def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    with tf.variable_scope(name_or_scope=scope_name, reuse=tf.AUTO_REUSE) as socpe:
        pool = tf.nn.max_pool(value=inputs, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)
    return pool

        
def fully_connected(inputs, out_dim, scope_name='fc'):
    ''' 
    A fully connected linear layer on inputs
    '''
    ####[input_channels,out_dim] 则input是按[batch_size,input_channesl]排列
    ####[out_dim,input_channels] 则input是按[input_channles,batch_size]排列
    with tf.variable_scope(name_or_scope=scope_name, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable(name='weights', shape=[inputs.shape[-1], out_dim], initializer=tf.truncated_normal_initializer) 
        b = tf.get_variable(name='biases', shape=[out_dim], initializer=tf.zeros_initializer)
        out = tf.add(tf.matmul(inputs, w), b)
    return out


def inception(inputs,filters,stride,padding,scope_name):
    with tf.variable_scope(name_or_scope=scope_name,reuse=tf.AUTO_REUSE) as scope:
        input_channels=inputs.shape[-1]
        kernel1_1=tf.get_variable(name='kernel1_1',shape=[1,1,input_channels,filters],initializer=tf.truncated_normal_initializer)

        kernel3_1=tf.get_variable(name='kernel3_1',shape=[1,1,input_channels,filters],initializer=tf.truncated_normal_initializer)

        kernel3_3=tf.get_variable(name='kernel3_3',shape=[3,3,filters,filters],initializer=tf.truncated_normal_initializer)

        kernel5_1=tf.get_variable(name='kernel5_1',shape=[1,1,input_channels,filters],initializer=tf.truncated_normal_initializer)
        
        kernel5_5=tf.get_variable(name='kernel5_5',shape=[5,5,filters,filters],initializer=tf.truncated_normal_initializer)

        kernel_max_1=tf.get_variable(name='conv_max_1',shape=[1,1,filters,filters],initializer=tf.truncated_normal_initializer)

        conv1_1 = tf.nn.conv2d(input=inputs, filter=kernel1_1, strides=[1, stride, stride, 1], padding=padding)
        conv3_1 = tf.nn.conv2d(input=inputs, filter=kernel3_1, strides=[1, stride, stride, 1], padding=padding)
        conv5_1 = tf.nn.conv2d(input=inputs, filter=kernel5_1, strides=[1, stride, stride, 1], padding=padding)
        
        conv3_3=tf.nn.conv2d(input=conv3_1,filter=kernel3_3,strides=[1,stride,stride,1],padding=padding)
        conv5_5=tf.nn.conv2d(input=conv5_1,filter=kernel5_5,strides=[1,stride,stride,1],padding=padding)

        max_pool1 = maxpool(inputs=conv5_5, ksize=3, stride=2, padding=padding, scope_name='max_pool1')
        conv_max_1=tf.nn.conv2d(input=max_pool1,filter=kernel_max_1,strides=[1,stride,stride,1],padding=padding)

        inception_conv=tf.concat(axis=3,values=[conv1_1,conv3_3,conv5_5])
        inception_depth=conv1_1.shape[-1]+conv3_3.shape[-1]+conv5_5.shape[-1]
        inception_biases=tf.get_variable(name='inception_biases',shape=[inception_depth],initializer=tf.zeros_initializer)
        out=tf.nn.relu(inception_conv) + inception_biases

        return out

class ConvNet(object):

    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.trainable = True
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000

####数据获取，使用dataset，得到iterator,从iterator中get batch
    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_mnist_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            # reshape the image to make it work with tf.nn.conv2d
            ####初始化initializer 初始化iterator
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)  # initializer for train_data

    def inference(self):
        '''
        Build the model according to the description we've shown in class
        '''
        inception1 = inception(inputs=self.img, filters=16,stride=1, padding='SAME', scope_name='inception1')
        inception2 = inception(inputs=inception1, filters=32,stride=1, padding='SAME', scope_name='inception2')

        max_pool1 = maxpool(inputs=inception2, ksize=2, stride=2, padding='VALID', scope_name='max_pool1')


        inception3 = inception(inputs=max_pool1, filters=64,stride=1, padding='SAME', scope_name='inception3')
        inception4 = inception(inputs=inception3, filters=128,stride=1, padding='SAME', scope_name='inception4')
        max_pool2 = maxpool(inputs=inception4, ksize=2, stride=2, padding='VALID', scope_name='max_pool2')

        batch=max_pool2.shape[0]
        pool=tf.reshape(tensor=max_pool2,shape=[-1,max_pool2.shape[1]*max_pool2.shape[2]*max_pool2.shape[3]])
        fc1 = fully_connected(inputs=pool, out_dim=1024, scope_name='fc1')
        self.logits = fully_connected(inputs=fc1, out_dim=10)

    def loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label))

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.variable_scope(name_or_scope='summary', reuse=tf.AUTO_REUSE) as scope:
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('loss', self.loss)
            self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/inception/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/inception')
        writer = tf.summary.FileWriter(logdir='d:/tensorboard/mnist/inception', graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/inception/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()


if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=10)