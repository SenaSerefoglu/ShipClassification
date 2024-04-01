import os
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, LearningRateScheduler



def get_data(directory, batch_size, img_size, shuffle=True, seed=42):
    data = tf.keras.utils.image_dataset_from_directory(directory,
                                                       batch_size=batch_size,
                                                       image_size=img_size,
                                                       shuffle=shuffle,
                                                       seed=seed)
    return data

def CNN_model():
    # Define the CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.8))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def compile_model(model):
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
def train_model(model, train_generator, test_generator, batch_size):
    # Implement learning rate schedule
    def lr_schedule(epoch):
        return 0.001 * (0.1 ** int(epoch / 10))

    learning_rate_scheduler = LearningRateScheduler(lr_schedule)


    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=20,  # Increase the number of epochs
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size,
        callbacks=[learning_rate_scheduler]
    )
    return history


def evaluate_model(model, test_generator):
    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    return loss, accuracy

def visualize_results(history):
    # Visualize the results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Define data directory
    train_dir = os.path.join(os.getcwd(), 'ships_dataset/train')
    test_dir = os.path.join(os.getcwd(), 'ships_dataset/test')

    # Define the image size and batch size
    img_size = (224, 224)  # You can adjust this based on your dataset
    batch_size = 32
    
    # Use ImageDataGenerator for data augmentation on the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches using train_datagen
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Flow validation images in batches using test_datagen
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Define the CNN model
    model = CNN_model()

    # Compile the model
    compile_model(model)

    # Train the model
    history = train_model(model, train_generator, test_generator, batch_size)

    # Evaluate the model
    loss, accuracy = evaluate_model(model, test_generator)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    # Visualize the results
    visualize_results(history)


    
if __name__ == '__main__':
    main()