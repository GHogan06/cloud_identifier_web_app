import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


#initialise directory, image size, batch size
dataset_dir = "cloud_dataset"
image_size = (400, 400)
batch_size = 16

# Load data
#training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)

#validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)

# Data Augmentation
#Applying random flipping, rotation, and zoom to stop the model from memorizing the training data
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

# Load the MobileNetV2 pre-trained model as a base for transfer learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(400, 400, 3),
    include_top=False,  # Exclude original ImageNet 1000-class classifier
    weights='imagenet'
)
base_model.trainable = False  # Freeze pre-trained weights so they don't get destroyed during early training

# Build Model Architecture using Functional API
inputs = tf.keras.Input(shape=(400, 400, 3))
x = data_augmentation(inputs)

# MobileNetV2 needs the pixel values preprocessed in the range [-1, 1]
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Pass through frozen base model
x = base_model(x, training=False)

# Convert feature maps to a 1D feature vector
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Final Dense layer for the 4 possible cloud classes
outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=["accuracy"],
)

# Train with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[early_stopping],
    epochs=30,
)

# ---------------------------------------------------------
# FINE-TUNING THE INITIAL MODEL (Gently update the top features)
# ---------------------------------------------------------

# Unfreeze base model
base_model.trainable = True

# Freeze all except the last 20 layers of the base model
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 3. Freeze BatchNormalization layers explicitly
for layer in base_model.layers[fine_tune_at:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Recompile with a much smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=["accuracy"]
)

# 5. Fine-tune for about 20 epochs
fine_tune_epochs = 20
total_epochs = len(history.epoch) + fine_tune_epochs
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=len(history.epoch),
    callbacks=[early_stopping]
)

# Save the model in Keras native format
model.save("cloud_identifier_mobilenet.keras")
print("Model successfully saved as cloud_identifier_mobilenet.keras!")

# Evaluating the model using confusion matrix
class_names = val_ds.class_names
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('MobileNetV2 Cloud Classification Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Cloud Type', fontsize=12)
plt.ylabel('Actual Cloud Type', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print Report
print("\nDetailed Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))