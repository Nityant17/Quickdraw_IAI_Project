import os
import shutil
import sys
from unittest.mock import MagicMock

# 1. Force the stable engines right at the start!
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 2. The Numpy 2.0+ fix
import numpy as np
np.object = object
np.bool = np.bool_
np.complex = complex

# 3. The Protobuf/Dependency Shields
sys.modules['tensorflow_hub'] = MagicMock()
sys.modules['tensorflow_decision_forests'] = MagicMock()

import tensorflow as tf
import tensorflowjs as tfjs

print("1. Loading your Keras 2 model...")
model = tf.keras.models.load_model('quickdraw_model.keras')

print("2. Converting directly to web format (No hacks needed!)...")
if not os.path.exists('web_model'):
    os.makedirs('web_model')

# Because we are using Keras 2, TFJS can convert it natively!
tfjs.converters.save_keras_model(model, 'web_model')

print("3. Moving classes.json...")
shutil.copy('quickdraw_model_classes.json', 'web_model/classes.json')

print("✅ SUCCESS! Start your server.")