import os

import tensorflow as tf
import math

import configuration
from inference_utils import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

import time


input_files = "/full/path/to/imageFile"
checkpoint_path = "saved_models/model.ckpt-3000000"
vocab_file = "saved_models/word_counts.txt"

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path",
                       checkpoint_path,
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")

tf.flags.DEFINE_string("vocab_file",
                       vocab_file,
                       "Text file containing the vocabulary.")

tf.flags.DEFINE_string("input_files",
                       input_files,
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")


def main(_):
  # Build the inference graph.
  start_time = time.time()
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for filename in filenames:
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
    print(time.time() - start_time)


if __name__ == "__main__":
  tf.app.run()
