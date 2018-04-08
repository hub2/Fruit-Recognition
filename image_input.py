import tensorflow as tf
import os


IMAGE_SIZE = 25
NUM_CLASSES = 7

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
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        float_image = tf.image.per_image_standardization(reshaped_image)

        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

        return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

