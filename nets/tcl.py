import tensorflow as tf
def arg_scope(func):
    def func_with_args(self, *args, **kwargs):
        if hasattr(self, 'pre_defined'):
            for k in self.pre_defined.keys():
                kwargs[k] = self.pre_defined[k]
            
        return func(self, *args, **kwargs)
    return func_with_args

class Conv2d(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, kernel_size, num_outputs, strides = 1, dilations = 1, padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(),
                 kernel_regularizer = None,
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 biases_regularizer = None,
                 activation_fn = tf.nn.relu):
        super(Conv2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        self.biases_regularizer = biases_regularizer
        
        self.activation_fn = activation_fn
        
        
    def build(self, input_shape):
        super(Conv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [int(input_shape[-1]), self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           regularizer = self.biases_regularizer)
        

    def call(self, input):
        conv = tf.nn.conv2d(input, self.kernel, self.strides, self.padding,
                            dilations=self.dilations, name=None)
        if self.use_biases:
            conv += self.biases
        if self.activation_fn:
            conv = self.activation_fn(conv)
        return conv
    
class FC(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, num_outputs, 
                 kernel_initializer = tf.keras.initializers.VarianceScaling(),
                 kernel_regularizer = None,
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 biases_regularizer = None,
                 activation_fn = tf.nn.relu):
        super(FC, self).__init__()
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        self.biases_regularizer = biases_regularizer
        
        self.activation_fn = activation_fn
        
        
    def build(self, input_shape):
        super(FC, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = [int(input_shape[-1]), self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           regularizer = self.biases_regularizer)
    def call(self, input):
        fc = tf.matmul(input, self.kernel)
        if self.use_biases:
            fc += self.biases
        if self.activation_fn:
            fc = self.activation_fn(fc)
        return fc

class BatchNorm(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, param_initializers = None,
                       param_regularizers = None,
                       scale = True,
                       center = True,
                       alpha = 0.9,
                       epsilon = 1e-5,
                       activation_fn = None):
        super(BatchNorm, self).__init__()
        if param_initializers == None:
            param_initializers = {}
        if not(param_initializers.get('moving_mean')):
            param_initializers['moving_mean'] = tf.keras.initializers.Zeros()
        if not(param_initializers.get('moving_stddev')):
            param_initializers['moving_stddev'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('gamma')) and scale:
            param_initializers['gamma'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('beta')) and center:
            param_initializers['beta'] = tf.keras.initializers.Zeros()
        
        if param_regularizers == None:
            param_regularizers = {}
            
        self.param_initializers = param_initializers
        self.param_regularizers = param_regularizers
        self.scale = scale
        self.center = center
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_fn = activation_fn
        
    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)
        self.moving_mean = self.add_weight(name  = 'moving_mean', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_mean'])
        self.moving_stddev = self.add_weight(name  = 'moving_stddev', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_stddev'])
        if self.scale:
            self.gamma = self.add_weight(name  = 'gamma', 
                                         shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                         initializer=self.param_initializers['gamma'],
                                         regularizer=self.param_regularizers.get('gamma'))
        if self.center:
            self.beta = self.add_weight(name  = 'beta', 
                                        shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                        initializer=self.param_initializers['beta'],
                                        regularizer=self.param_regularizers.get('beta'))
            
    def EMA(self, variable, value):
        update_delta = (variable - value) * self.alpha
        variable.assign(variable-update_delta)
        
    def call(self, input, training=None):
        if training:
            mean, var = tf.nn.moments(input, list(range(len(input.get_shape())-1)))
            stddev = tf.sqrt(var + self.epsilon)
            bn = (input-mean)/stddev
            self.EMA(self.moving_mean, mean)
            self.EMA(self.moving_stddev, stddev)
        else:
            bn = (input-self.moving_mean)/self.moving_stddev
        if self.scale:
            bn *= self.gamma
        if self.center:
            bn += self.beta
        if self.activation_fn:
            bn = self.activation_fn(bn)
        return bn
    
    
#@tf.custom_gradient
#def custom_op(x):
#    result = ... # do forward computation
#    def custom_grad(dy):
#        grad = ... # compute gradient
#        return grad
#    return result, custom_grad