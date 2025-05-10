import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# # Paths
# base_dir = 'Dataset'
# train_dir = os.path.join(base_dir, 'Training')
# test_dir = os.path.join(base_dir, 'Testing')

# # EDA: count images per class
# def count_images(folder):
#     class_counts = {}
#     for cls in os.listdir(folder):
#         cls_path = os.path.join(folder, cls)
#         if os.path.isdir(cls_path):
#             count = len(os.listdir(cls_path))
#             class_counts[cls] = count
#     return class_counts

# train_counts = count_images(train_dir)
# test_counts = count_images(test_dir)

# print("Train counts:", train_counts)
# print("Test counts:", test_counts)

# # Plot class distribution
# sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()))
# plt.title('Training Set Class Distribution')
# plt.show()

# # Show sample images
# fig, axes = plt.subplots(1, 4, figsize=(15, 5))
# for i, cls in enumerate(os.listdir(train_dir)):
#     img_path = os.path.join(train_dir, cls, os.listdir(os.path.join(train_dir, cls))[0])
#     img = Image.open(img_path)
#     axes[i].imshow(img)
#     axes[i].set_title(cls)
#     axes[i].axis('off')
# plt.show()


# # Preprocessing + Data Augmentation
# img_size = (224, 224)
# batch_size = 32

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2  # 80% train, 20% validation
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_gen = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# val_gen = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

# test_gen = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=False
# )

# # Model: EfficientNetV2
# base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# base_model.trainable = False

# x = GlobalAveragePooling2D()(base_model.output)
# x = Dense(128, activation='relu')(x)
# output = Dense(train_gen.num_classes, activation='softmax')(x)

# model = Model(base_model.input, output)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.summary()

# # Train the model
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=20,
#     callbacks=[early_stop]
# )

# # Save the model
# model.save('brain_tumor_model.h5')
# print("Model saved as brain_tumor_model.h5")

# # Plot training history
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Acc')
# plt.plot(history.history['val_accuracy'], label='Val Acc')
# plt.title('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Loss')
# plt.legend()

# plt.show()

# # Evaluate on test set
# test_loss, test_acc = model.evaluate(test_gen)
# print(f"Test Accuracy: {test_acc:.4f}")

# # Predict on test set
# preds = model.predict(test_gen)
# y_pred = np.argmax(preds, axis=1)
# y_true = test_gen.classes
# class_labels = list(test_gen.class_indices.keys())

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6, 6))
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()

# # Classification report
# report = classification_report(y_true, y_pred, target_names=class_labels)
# print("Classification Report:\n", report)