import tensorflow as tf
import numpy as np

from dataloader import Dataloader
import op_util, tcl

### define path and hyper-parameter
Learning_rate =1e-1
batch_size = 128
val_batch_size = 200
train_epoch = 100
weight_decay = 5e-4

should_log          = 100
save_summaries_secs = 20
gpu_num = '0'

train_images, train_labels, val_images, val_labels, pre_processing = Dataloader('cifar10', '', '')
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(val_batch_size)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tcl.Conv2d([3,3], 32, 
                                kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                use_biases = False,
                                activation_fn = None)
        self.bn1 = tcl.BatchNorm(activation_fn = tf.nn.relu)
        
        self.d1 = tcl.FC(128, kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                         activation_fn=tf.nn.relu)
        self.d2 = tcl.FC(10, kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                         activation_fn=tf.nn.softmax)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)    
        x = tf.reduce_mean(x,[1,2])
        x = self.d1(x)
        return self.d2(x)

model = MyModel()
train_step, train_loss, train_accuracy, test_step, test_loss, test_accuracy = op_util.Optimizer(model, Learning_rate)
summary_writer = tf.summary.create_file_writer('tmp/summaries')

model(np.zeros([1]+list(train_images.shape[1:])), training = True)

for epoch in range(train_epoch):
    with summary_writer.as_default():
        for images, labels in train_ds:
            images = pre_processing(images, is_training = True)
            train_step(images, labels)
        tf.summary.scalar('Categorical_loss/train', train_loss.result(), step=epoch)
        tf.summary.scalar('Accuracy/train', train_accuracy.result(), step=epoch)
            
        for test_images, test_labels in test_ds:
            images = pre_processing(images, is_training = False)
            test_step(test_images, test_labels)
        tf.summary.scalar('Categorical_loss/test', test_loss.result(), step=epoch)
        tf.summary.scalar('Accuracy/test', test_accuracy.result(), step=epoch)
            
    template = 'Epoch: {0:3d}, train_loss: {1:0.4f}, train_Acc.: {2:2.2f}, val loss: {3:0.4f}, val_Acc.: {4:2.2f}'
    print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                                     test_loss.result(),  test_accuracy.result()*100))
