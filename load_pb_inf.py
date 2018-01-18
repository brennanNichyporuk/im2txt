import tensorflow as tf  # Default graph is initialized when the library is imported
from tensorflow.python.platform import gfile
import time
import math

from inference_utils import vocabulary
from inference_utils import utils


FLAGS = tf.flags.FLAGS
vocab_file = "saved_models/word_counts.txt"
tf.flags.DEFINE_string("vocab_file",
                       vocab_file,
                       "Text file containing the vocabulary.")

input_files = "/full/path/to/ImageFile"
tf.flags.DEFINE_string("input_files",
                       input_files,
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

pb_file = "saved_models/model.pb"
vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

start_time = time.time()
# Reference: https://github.com/MarvinTeichmann/KittiSeg/issues/113
with tf.Graph().as_default() as graph:  # Set default graph as graph

    with tf.Session() as sess:
        # Load the graph in graph_def
        print("load graph")

        # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
        with gfile.FastGFile(pb_file, 'rb') as f:

            # Set FCN graph to the default graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            # Import a graph_def into the current default Graph
            # (In this case, the weights are (typically) embedded in the graph)
            tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="",
                    op_dict=None,
                    producer_op_list=None
                    )

            filenames = []
            for file_pattern in FLAGS.input_files.split(","):
                filenames.extend(tf.gfile.Glob(file_pattern))

            with tf.gfile.GFile(filenames[0], "r") as f:
                image = f.read()

            #initialize_all_variables
            tf.global_variables_initializer()

            captions = utils.beam_search(sess, image, vocab)
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

print(time.time() - start_time)
