import tensorflow as tf
import numpy as np
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataloader import Dataloader
import op_util
from nets import ResNet

### define path and hyper-parameter
train_path = 'tmp/summaries'
Learning_rate =1e-1
batch_size = 128
val_batch_size = 200
train_epoch = 100
weight_decay = 5e-4

should_log          = 100
save_summaries_secs = 20
gpu_num = 0

tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')

summary_writer = tf.summary.create_file_writer(train_path)

train_images, train_labels, val_images, val_labels, pre_processing = Dataloader('cifar10', '', '')
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(val_batch_size)

model = ResNet.Model(num_layer = 18, weight_decay = weight_decay, num_class = np.max(train_labels)+1)
LRS = op_util.learning_rate_scheduler(Learning_rate, train_epoch, [0.3, 0.6, 0.8], 0.2)

tf.summary.trace_on(graph=True, profiler=True)
train_step, train_loss, train_accuracy, test_step, test_loss, test_accuracy = op_util.Optimizer(model, LRS.LR)
summary_writer = tf.summary.create_file_writer('tmp/summaries')

model(np.zeros([1]+list(train_images.shape[1:])), training = True)

with summary_writer.as_default():
    step = 0
    train_time = time.time()
    for epoch in range(train_epoch):
        LRS(epoch)
        for images, labels in train_ds:
            images = pre_processing(images, is_training = True)
            train_step(images, labels)
            step += 1
            if step % should_log == 0:
                template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                print (template.format(step, train_loss.result(), (time.time()-train_time)/should_log))
                train_time = time.time()
        tf.summary.scalar('Categorical_loss/train', train_loss.result(), step=epoch+1)
        tf.summary.scalar('Accuracy/train', train_accuracy.result()*100, step=epoch+1)
            
        for test_images, test_labels in test_ds:
            test_images = pre_processing(test_images, is_training = False)
            test_step(test_images, test_labels)
        tf.summary.scalar('Categorical_loss/test', test_loss.result(), step=epoch+1)
        tf.summary.scalar('Accuracy/test', test_accuracy.result()*100, step=epoch+1)
            
        template = 'Epoch: {0:3d}, train_loss: {1:0.4f}, train_Acc.: {2:2.2f}, val loss: {3:0.4f}, val_Acc.: {4:2.2f}'
        print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                                         test_loss.result(),  test_accuracy.result()*100))

