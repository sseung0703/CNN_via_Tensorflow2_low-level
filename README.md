# CNN_via_TF2_low-level-API
This project's purpose is building convolutional neural network implementation with Tensorflow2.0 low-level API only. It looks somewhat useless things because Keras layers are so easy to handle. However, I have many layers that were built in TF1 low-level API, and surely, there is no Keras layer for them. So, I have to do this, and perhaps some guys need it too.

# Require
- Tensorflow > 2.0

# Contents
- tcl.py
    - Build a custom layer and add trainable or untrainable parameters.
    - Add regularization for each trainable parameters.
    - Update function for the moving mean and the moving standard deviation.
    - conditioning for training and inference phase.
    
- op_util.py
    - define a loss function with regularization losses.
    - build optimizer with computing and applying gradients.
    - define steps for training and inference.
    
- train_and_validate.py
    - load dataset, model, and optimizer.
    - do train and validate.
    - visualize the log via Tensorboard.

# To do
- write Readme and milestones.
- codes to save and load models without a checkpoint.
- make the example networks such as ResNet, WResNet.
- find more things to do...
