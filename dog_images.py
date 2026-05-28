
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
data_dir = "C:/Users/Harsh/PycharmProjects/dogbreed_imageclassification/dog_images"

batch_size = 16
img_height = 224
img_width = 224

train_dataset, validation_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    subset='both',
    seed=123,
    shuffle=True,
)

print("Class names:", train_dataset.class_names)
print("Training samples:", len(train_dataset) * batch_size)
print("Validation samples:", len(validation_dataset) * batch_size)


# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Correct way to get one image
#for images, labels in train_dataset.take(1):
#    image = images[0]      # First image in the batch
#    label = labels[0]      # Its label
#    break
#
## Display
#plt.figure(figsize=(6, 6))
#plt.imshow(image.numpy().astype("uint8"))
#plt.title(f"Label: {label.numpy()}")
#plt.axis("off")
#plt.show()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1,.1),
    tf.keras.layers.RandomContrast(0.2),

])

base_model = ResNet50(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet",
   )

base_model.trainable = False

model = models.Sequential([
    base_model,
    data_augmentation,
    layers.Rescaling(1./255,),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(5, activation="softmax"),
])



model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
)

print("training completed")
model.save('dog_breed_model.h5')
print("model saved")