import tensorflow as tf
from nets import tcl

class Model(tf.keras.Model):
    def __init__(self, architecture, weight_decay, num_class):
        super(Model, self).__init__(architecture, weight_decay, num_class)
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                                  use_biases = False, activation_fn = None))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(param_regularizers = {'gamma':tf.keras.regularizers.l2(weight_decay),
                                                                           'beta':tf.keras.regularizers.l2(weight_decay)}))
        setattr(tcl.FC, 'pre_defined', kwargs(kernel_regularizer = tf.keras.regularizers.l2(weight_decay),
                                              biases_regularizer = tf.keras.regularizers.l2(weight_decay)))
        
        self.wresnet_layers = {}
        depth, widen_factor = architecture
        self.nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.stride = [1,2,2]
        self.n = (depth-4)//6
        self.net_name = 'WResNet'
        self.feature = []
        self.feature_noact = []
        self.last_embedded = []
        self.logits = []
        
        with tf.name_scope(self.net_name):
            self.wresnet_layers[self.net_name + '/conv0'] = tcl.Conv2d([3,3], self.nChannels[0])
            self.wresnet_layers[self.net_name + '/bn0']   = tcl.BatchNorm(activation_fn = tf.nn.relu)
            in_planes = self.nChannels[0]
            
            for i, (c, s) in enumerate(zip(self.nChannels[1:], self.stride)):
                for j in range(self.n):
                    block_name = '/BasicBlock%d.%d'%(i,j)
                    with tf.name_scope(block_name[1:]):
                        equalInOut = in_planes == c
                        nb_name = self.net_name + block_name
                        
                        self.wresnet_layers[nb_name + '/bn0']   = tcl.BatchNorm()
                        self.wresnet_layers[nb_name + '/conv1'] = tcl.Conv2d([3,3], c, strides = s if j == 0 else 1)
                        self.wresnet_layers[nb_name + '/bn1']   = tcl.BatchNorm(activation_fn = tf.nn.relu)
                        self.wresnet_layers[nb_name + '/drop']  = tcl.Dropout(0.7)
                        self.wresnet_layers[nb_name + '/conv2'] = tcl.Conv2d([3,3], c, strides = 1)
                                
                        if not(equalInOut):
                            self.wresnet_layers[nb_name + '/conv3'] = tcl.Conv2d([1,1], c, strides = s if j == 0 else 1)
                        in_planes = c
            self.wresnet_layers[self.net_name + '/bn1']   = tcl.BatchNorm()
            self.wresnet_layers['FC'] = tcl.FC(num_class)
    
    def call(self, x, training=None):
        with tf.name_scope(self.net_name):
            x = self.wresnet_layers[self.net_name + '/conv0'](x)
            x = self.wresnet_layers[self.net_name + '/bn0'](x, training = training)
            in_planes = self.nChannels[0]
            
            for i, (c, s) in enumerate(zip(self.nChannels[1:], self.stride)):
                for j in range(self.n):
                    block_name = '/BasicBlock%d.%d'%(i,j)
                    with tf.name_scope(block_name[1:]):
                        equalInOut = in_planes == c
                        nb_name = self.net_name + block_name
                        if not equalInOut:
                            x_ = self.wresnet_layers[nb_name + '/bn0'](x, training = training)
                            x = tf.nn.relu(x_)
                            if j == 0 and i > 0:
                                self.feature.append(x)
                                self.feature_noact.append(x_)
                        else:
                            out_ = self.wresnet_layers[nb_name + '/bn0'](x, training = training)
                            out = tf.nn.relu(out_)
                            if j == 0 and i > 0:
                                self.feature.append(out)
                                self.feature_noact.append(out_)
                            
                        out = self.wresnet_layers[nb_name + '/conv1'](out if equalInOut else x)
                        out = self.wresnet_layers[nb_name +   '/bn1'](out, training = training)
#                        out = self.wresnet_layers[nb_name +  '/drop'](out, training = training)
                        out = self.wresnet_layers[nb_name + '/conv2'](out)
                        if not(equalInOut):
                            x = self.wresnet_layers[nb_name + '/conv3'](x)
                        x = x+out
                        in_planes = c
                        
            x_ = self.wresnet_layers[self.net_name + '/bn1'](x, training = training)
            x = tf.nn.relu(x_)
            self.feature.append(x)
            self.feature_noact.append(x_)
                            
            x = tf.reduce_mean(x,[1,2])
            self.last_embedded.append(x)
            x = self.wresnet_layers['FC'](x)
            self.logits.append(x)
            return x
        
    def get_feat(self, x, feat):
        self.call(x, True)
        return getattr(self, feat)
