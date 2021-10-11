import matplotlib.pyplot as plt
import tensorflow as tf

def create_adversarial_pattern(input_images, input_labels, model):
  loss_object = tf.keras.losses.CategoricalCrossentropy()

  with tf.GradientTape() as tape:
    tape.watch(input_images)
    prediction = model(input_images, training = False)
    loss = loss_object(input_labels, prediction)

  # Get the gradients of the loss w.r.t to the input images
  gradient = tape.gradient(loss, input_images)

  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.math.sign(gradient)
  return signed_grad

def generate_adversarial_images(input_images, input_labels, epsilon, model):
  input_images = tf.convert_to_tensor(input_images)
  perturbations = create_adversarial_pattern(input_images, input_labels, model)
  adversarial_images = input_images + epsilon * perturbations
  adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
  return adversarial_images
