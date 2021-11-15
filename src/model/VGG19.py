import tensorflow as tf

def vgg_module(conv_layers, filters, kernel_size, padding, activation):
  list_of_layers = []
  # CNN Layers
  for i in range(conv_layers):
    list_of_layers.append(tf.keras.layers.Conv2D(
      filters = filters,
      kernel_size = kernel_size,
      padding = padding,
      activation = activation
      )
    )
    list_of_layers.append(tf.keras.layers.BatchNormalization())

  # Pooling Layer
  list_of_layers.append(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
  return list_of_layers

def VGG19(input_shape, num_classes):
  layers = []
  input = tf.keras.layers.Input(shape = input_shape)
  layers.append(input)

  # Block 1
  block_1 = vgg_module(2, 64, (3,3), 'same', tf.nn.relu)
  layers += block_1

  # Block 2
  block_2 = vgg_module(2, 128, (3,3), 'same', tf.nn.relu)
  layers += block_2

  # Block 3
  block_3 = vgg_module(4, 256, (3,3), 'same', tf.nn.relu)
  layers += block_3

  # Block 4
  block_4 = vgg_module(4, 512, (3,3), 'same', tf.nn.relu)
  layers += block_4

  # Block 5
  block_5 = vgg_module(4, 512, (3,3), 'same', tf.nn.relu)
  layers += block_5

  # Block 6
  block_6 = [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation = tf.nn.relu),
    tf.keras.layers.Dense(4096, activation = tf.nn.relu),
    tf.keras.layers.Dense(num_classes, activation = tf.nn.softmax)
  ]
  layers += block_6

  model = tf.keras.Sequential(layers = layers)
  return model
