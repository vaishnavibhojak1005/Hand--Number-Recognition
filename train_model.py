import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Directory where your images are stored
train_dir = 'gesture_images'  # Update with the correct path to your dataset

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale image pixels to the range [0, 1]
    rotation_range=40,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Randomly apply shearing transformations
    zoom_range=0.2,  # Randomly zoom images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest',  # Filling missing pixels after transformations
    # Additional augmentation:
    brightness_range=[0.5, 1.5],  # Random brightness
    channel_shift_range=30.0,  # Random color jitter
)

# Image data loading with class labels from folder names
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to your training data
    target_size=(224, 224),  # Resize to match input size of the model
    batch_size=32,  # Batch size for training
    class_mode='categorical'  # Multi-class classification
)

# Print out class indices to check class labeling
print("Class indices:", train_generator.class_indices)

# Load the base MobileNetV2 model without the top layer (we'll add our custom classifier on top)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model to prevent them from being trained
base_model.trainable = False

# Build the full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Global average pooling layer
    layers.Dense(1024, activation='relu'),  # Fully connected layer
    layers.Dropout(0.5),  # Dropout layer for regularization
    layers.Dense(5, activation='softmax')  # Output layer with 10 classes (change if you have more/less)
])

# Compile the model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']
)

# Train the model
try:
    model.fit(
        train_generator,
        epochs=10,
        verbose=2  # Increased verbosity to see more details during training
    )
except ValueError as e:
    print("Error:", e)
    print("Please check if the directory structure is correct and images are present.")

# Save the trained model
model.save('gesture_model.h5')
print("Model saved as 'gesture_model.h5'")
