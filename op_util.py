import tensorflow as tf

def Optimizer(model, LR):
    with tf.name_scope('Optimizer_w_Distillation'):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(LR, .9, nesterov=True)
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
    @tf.function
    def training(images, labels, lr):
        with tf.GradientTape() as tape:
            predictions = model(images, training = True)
            loss = loss_object(labels, predictions)
            regularizer_loss = tf.add_n(model.losses)
            total_loss = loss + regularizer_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.learning_rate.assign(lr)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_accuracy(labels, predictions)
        
    @tf.function
    def validation(images, labels):
        predictions = model(images, training = False)
        loss = loss_object(labels, predictions)
        
        test_loss(loss)
        test_accuracy(labels, predictions)
    return training, train_loss, train_accuracy, validation, test_loss, test_accuracy

class learning_rate_scheduler(tf.keras.layers.Layer):
    def __init__(self, init_lr, total_epoch, decay_points, decay_rate):
        super(learning_rate_scheduler, self).__init__()
        self.init_lr = init_lr
        self.total_epoch = total_epoch
        self.decay_points = decay_points
        self.decay_rate = decay_rate
        self.current_lr = init_lr
        
    def call(self, epoch):
        with tf.name_scope('learning_rate_scheduler'):
            Learning_rate = self.init_lr
            for i, dp in enumerate(self.decay_points):
                Learning_rate = tf.cond(tf.greater_equal(epoch, int(self.total_epoch*dp)), lambda : Learning_rate*self.decay_rate,
                                                                                           lambda : Learning_rate)
            self.current_lr = Learning_rate
            return Learning_rate
            
            
