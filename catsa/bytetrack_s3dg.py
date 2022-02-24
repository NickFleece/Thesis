import s3dg
import numpy as np
import tensorflow as tf

print(
        s3dg.s3dg(
            tf.convert_to_tensor(
                np.ones((16,10,10,10,3)), dtype=tf.float32
            )
        )[0].shape
)