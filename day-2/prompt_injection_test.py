import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Load dataset correctly
dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
train_data, test_data = dataset['train'], dataset['test']

# Normalize the data
def normalize_data(dataset):
    return dataset.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl))

train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

# Convert the dataset into arrays of features and labels
x_train = []
y_train = []
for image, label in train_data:
    x_train.append(image.numpy())
    y_train.append(label.numpy())

x_train = np.array(x_train)
y_train = np.array(y_train)

# Now, do the same for x_test and y_test
x_test = []
y_test = []
for image, label in test_data:
    x_test.append(image.numpy())
    y_test.append(label.numpy())

x_test = np.array(x_test)
y_test = np.array(y_test)

# Check the shape
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Build model
model = Sequential([
    layers.InputLayer(input_shape=(28, 28)),  # Using InputLayer
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)  # 10 output classes for the MNIST dataset
])

# Compile model
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5)

# Generate adversarial examples using FGSM
def create_adversarial_pattern(model, image, label):
    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor(label)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image, training=False)
        loss = tf.losses.sparse_categorical_crossentropy(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad

# Generate adversarial example for a single image
image = x_test[0:1]
label = y_test[0:1]
perturbations = create_adversarial_pattern(model, image, label)
epsilon = 0.7  # You can test with 0.4, 0.5 if needed
adversarial_example = image + epsilon * perturbations
adversarial_example = tf.clip_by_value(adversarial_example, 0.0, 1.0)
# Adjust the perturbation strength


# Visual comparison between original and adversarial image
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image[0], cmap='gray') 

plt.subplot(1, 2, 2)
plt.title("Adversarial")
plt.imshow(adversarial_example[0], cmap='gray')

plt.show()

# Compare model predictions on original and adversarial image
orig_pred = model.predict(image)
adv_pred = model.predict(adversarial_example)

print("Original Label Prediction:", np.argmax(orig_pred))
print("Adversarial Label Prediction:", np.argmax(adv_pred))
print("Actual Clean Label:", label[0])


#print("Clean Label:", label.numpy[0])
#print("Clean Prediction:", np.argmax(orig_pred))
#print("Adversarial Prediction:", np.argmax(adv_pred))

#print("Original prediction: ", np.argmax(orig_pred))
#print("Adversarial prediction: ", np.argmax(adv_pred))


# Visualize adversarial example
#plt.imshow(adversarial_example[0], cmap='gray')
#plt.show()

# Make a prediction on the adversarial example
#adv_pred = model.predict(adversarial_example)
#print("Predicted label for adversarial example: ", np.argmax(adv_pred))

# Make a prediction on the original clean image
#clean_pred = model.predict(image)
#print("Predicted label for clean image: ", np.argmax(clean_pred))

