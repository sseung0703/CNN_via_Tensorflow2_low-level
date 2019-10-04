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
            

def learning_rate_scheduler(Learning_rate, epochs, decay_point, decay_rate):
    with tf.variable_scope('learning_rate_scheduler'):
        e, ie, te = epochs
        for i, dp in enumerate(decay_point):
            Learning_rate = tf.cond(tf.greater_equal(e, ie + int(te*dp)), lambda : Learning_rate*decay_rate, 
                                                                          lambda : Learning_rate)
        tf.summary.scalar('learning_rate', Learning_rate)
        return Learning_rate