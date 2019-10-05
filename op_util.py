import tensorflow as tf

def Optimizer(model, LR):
    with tf.name_scope('Optimizer_w_Distillation'):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD(LR, .9, nesterov=True)
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training = True)
                loss = loss_object(labels, predictions)
                regularizer_loss = tf.add_n(model.losses)
                total_loss = loss + regularizer_loss
                
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(labels, predictions)
        
        @tf.function
        def test_step(images, labels):
            predictions = model(images, training = False)
            loss = loss_object(labels, predictions)
            
            test_loss(loss)
            test_accuracy(labels, predictions)
        return train_step, train_loss, train_accuracy, test_step, test_loss, test_accuracy
            
class learning_rate_scheduler(tf.keras.layers.Layer):
    def __init__(self, Learning_rate, train_epoch, decay_point, decay_rate):
        super(learning_rate_scheduler, self).__init__()
        self.train_epoch = train_epoch
        self.decay_point = decay_point
        self.decay_rate = decay_rate
        self.LR = self.add_weight(name  = 'learning_rate', trainable = False, shape = [],
                                  initializer=tf.constant_initializer(Learning_rate))
    
    def call(self, epoch):
        super(learning_rate_scheduler, self).call(epoch)
        epoch = tf.cast(epoch, tf.float32)
        for i, dp in enumerate(self.decay_point):
            LR = tf.cond(tf.greater_equal(epoch, self.train_epoch*dp), lambda : self.LR*self.decay_rate, 
                                                                       lambda : self.LR)
        self.LR.assign(LR)
        return self.LR
