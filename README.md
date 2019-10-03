# CNN_via_Tensorflow2_low-level
This project's purpose is building convolutional neural network implementation with Tensorflow2.0 low-level API only. It looks somewhat useless things because Keras layers are so easy to handle. However, I have many layers that were built in TF1 low-level API, and surely, there is no Keras layer for them. So, I have to do this, and perhaps some guys need it too.

# Require
- Tensorflow > 2.0

# Contents
- tcl.py
    - Build a custom layer and add trainable or untrainable parameters.
    - Add regularization for each trainable parameters.
    - Define update function for the moving mean and the moving standard deviation.
    - Conditioning for training and inference phase.
    
- op_util.py
    - Define a loss function with regularization losses.
    - Build optimizer with computing and applying gradients.
    - Define steps for training and inference.
    
- train_and_validate.py
    - Load dataset, model, and optimizer.
    - Do train and validate.
    - Visualize the log via Tensorboard.

# To do
- Write Readme and milestones.
- Codes to save and load models without a checkpoint.
- Make the example networks such as ResNet, WResNet.
- Find more things to do...
