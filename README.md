# im2txt
This repo is a way to run inference on the [im2txt tensorflow project](https://github.com/tensorflow/models/tree/master/research/im2txt). Ask me for the pre-trained ckpt file.

# Dependencies
The caveat is it can only run on Tensorflow r0.12 because of certain changes to layer names since it was trained. 
Tensorflow r0.12 can be installed following the instructions [here](https://www.tensorflow.org/versions/r0.12/get_started/os_setup).


# To-Do
Still want to add a section to freeze the graph and convert to .pb format, allowing easy inference on mobile devices.
