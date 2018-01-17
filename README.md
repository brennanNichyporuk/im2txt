# im2txt
This repo is a way to run inference on the [im2txt tensorflow project](https://github.com/tensorflow/models/tree/master/research/im2txt). Ask me for the pre-trained ckpt file.

# Dependencies
The caveat is it can only run on Tensorflow r0.12 because of certain changes to layer names since it was trained. 
Tensorflow r0.12 can be installed following the instructions [here](https://www.tensorflow.org/versions/r0.12/get_started/os_setup).

# Running
To run inference on an image (or group of images), simply open [run_inference.py](https://github.com/roggirg/im2txt/blob/master/run_inference.py), modify the variables "input_files", "checkpoint_path" and "vocab_file" accordingly, and run the script.

# To-Do
Still want to add a section to freeze the graph and convert to .pb format, allowing easy inference on mobile devices.
