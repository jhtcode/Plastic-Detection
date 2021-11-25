# Plastic-Detection
Detecting plastic contaminant in seed cotton (raw cotton). Neural network model is built to detect plastic in cotton. I tried transfer learning method and also built NN model from scratch.
the nn notebook contains the NN model building and training.
plastic_eliminator is the python program for detecting and removing plastic contaminant. 
The NN model was converted to TensorRT format for faster inference, The target device for deployment is Jetson Nano 2GB, and I can get 24 fps with my model, but only 12 fps with transfer learning from MobileNetV2

