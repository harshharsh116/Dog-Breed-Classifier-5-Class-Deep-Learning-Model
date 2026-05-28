from tensorflow.keras.models import load_model


model = load_model("dog_breed_model.h5")


model.save("dog_breed_model.keras")

print("Model converted successfully!")