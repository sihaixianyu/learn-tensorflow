import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()

    tensor_1 = tf.constant([1, 2, 3])
    tensor_2 = tf.constant([4, 5, 6])
    tensor_3 = tf.concat([tensor_1, tensor_2], axis=0)

    g_initializer = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(g_initializer)
        res = sess.run(tensor_3)
        print(res)
