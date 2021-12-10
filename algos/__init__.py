# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms

from algos.acre.acre import acre as acre

# Loggers
from utils.logx import Logger, EpochLogger

# Version
from algos.version import __version__