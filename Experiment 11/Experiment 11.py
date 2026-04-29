import tensorflow as tf
import matplotlib.pyplot as plt

image_path = "C:\\Users\\DELL\\OneDrive\\Desktop\\intro neural networks\\sample 1.jpeg"

# Loading and processing the image
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, [300, 300]) # Resizing for the model
image = tf.image.convert_image_dtype(image, tf.float32)

print("Original Image Shape:", image.shape)

# Visualizing
plt.figure(figsize=(5,5))
plt.imshow(tf.squeeze(image))
plt.title("Original Image")
plt.axis('off')
plt.show()

# Adding batch dimension (required for CNN input)
image = tf.expand_dims(image, axis=0)