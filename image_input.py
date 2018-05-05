import tensorflow as tf
import os


IMAGE_SIZE = 100
NUM_CLASSES = 42

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000



def read_image(filename_queue):
    class ImageRecord:
        pass
    result = ImageRecord()
    label_bytes = 1
    result.width = IMAGE_SIZE
    result.height = IMAGE_SIZE
    result.depth = 3
    image_bytes = result.width * result.height * result.depth

    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width]
    )
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1,2,0])

    return result
def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'batch_1.bin')]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_image(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE//2
    width = IMAGE_SIZE//2

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle=False):
    # shuffle not implemented lulz
    num_preprocess_threads = 8

    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image("images", images)

    return images, tf.reshape(label_batch, [batch_size])


def inputs(eval_data, data_dir, batch_size):

    if not eval_data:
        filenames = [os.path.join(data_dir, 'batch_1.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join('ValidationBatches/batch_1.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("no nie ma plika")

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames)
        read_input = read_image(filename_queue)

        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        height = IMAGE_SIZE//2
        width = IMAGE_SIZE//2

        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)
        float_image = tf.image.per_image_standardization(resized_image)

        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

        return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

