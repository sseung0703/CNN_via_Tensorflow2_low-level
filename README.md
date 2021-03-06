# CNN_via_Tensorflow2_low-level
This project's purpose is building convolutional neural network implementation with Tensorflow2.0 low-level API only. It looks somewhat useless things because Keras layers are so easy to handle. However, I have many layers that were built in TF1 low-level API, and surely, there is no Keras layer for them. So, I have to do this, and perhaps some guys need it too.

# Require
- Tensorflow > 2.0

# Contents
- nets/tcl.py
    - Build a custom layer and add trainable or untrainable parameters.
    - Add regularization for each trainable parameters.
    - Define update function for the moving mean and the moving standard deviation.
    - Conditioning for training and inference phase.
    - Prototype of arg_scope. (will be updated)
    
- op_util.py
    - Define a loss function with regularization losses.
    - Build optimizer with computing and applying gradients.
    - Define steps for training and inference.
    - Learning rate scheduler
    
- train_and_validate.py
    - Load dataset, pre_processing algorithn, model, and optimizer.
    - Do train and validate.
    - Visualize the log via Tensorboard.
    
- dataloader.py
    - Load dataset
    - Define pre-processing algorithm.
    
- nets/ResNet.py and WResNet.py
    - Build a custom model via custom layers.
    - How to use implemented arg_scope

# To do
- Write Readme and milestones.
- Codes to save and load models without a checkpoint.
- Improve readability of a custom model.
- Find more things to do...
