# im2txt
This repo is a way to run inference on the [im2txt tensorflow project](https://github.com/tensorflow/models/tree/master/research/im2txt). Ask me for the pre-trained .ckpt and/or .pb file.

# Dependencies
The caveat to using the ckpt file is it can only run on Tensorflow r0.12 because of certain changes to layer names since it was trained. Tensorflow r0.12 can be installed following the instructions [here](https://www.tensorflow.org/versions/r0.12/get_started/os_setup).

Using the pb file, you can run on any tensorflow version (latest version tested with: Tensorflow r1.4.0).

# Running
To run inference on an image (or group of images) using the ckpt file:
1. Download the ckpt file and place it in the [saved_models](https://github.com/roggirg/im2txt/tree/master/saved_models) directory. 
2. Open [run_inference.py](https://github.com/roggirg/im2txt/blob/master/run_inference.py), modify the variable "input_files" pointing to the full path to your image (or images).
3. Run the script.

To run inference on an image (or group of images) using the pb file:
1. Download the pb file and place it in the [saved_models](https://github.com/roggirg/im2txt/tree/master/saved_models) directory. 
2. Open [load_pb_inf.py](https://github.com/roggirg/im2txt/blob/master/load_pb_inf.py), modify the variable "input_files" pointing to the full path to your image (or images).
3. Run the script.
