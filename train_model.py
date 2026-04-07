import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
import glob
import numpy as np

# THE Hack: Tricks older tensorflowjs into working with the newest NumPy
np.object = object
np.bool = np.bool_
np.complex = complex

class QuickDrawNumpyTrainer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.img_size = 28
        self.classes = []
        self.model = None
        self.num_classes = 0
    
    def load_data(self, max_samples_per_class=12000):
        """Loads .npy files directly from the data folder"""
        print(f"\nLooking for .npy files in {self.data_dir}/...")
        
        files = glob.glob(os.path.join(self.data_dir, "*.npy"))
        
        if not files:
            raise FileNotFoundError(f"❌ No .npy files found in {self.data_dir}/. Please download them from Google QuickDraw!")
        
        # The class name is just the filename (e.g., 'apple.npy' -> 'apple')
        # This automatically deletes the annoying Google prefix!
        self.classes = [os.path.splitext(os.path.basename(f))[0].replace('full_numpy_bitmap_', '') for f in files]
        self.num_classes = len(self.classes)
        
        print(f"Found {self.num_classes} classes: {self.classes}")
        
        x_data = []
        y_data = []
        
        print("\nLoading images into memory...")
        for i, file_path in enumerate(files):
            print(f"  Loading {self.classes[i]}...")
            
            # Load the numpy array
            data = np.load(file_path)
            
            # QuickDraw files have 100,000+ images per class. 
            # We slice it to avoid crashing your RAM!
            data = data[:max_samples_per_class]
            
            # Google stores them as a flat list of 784 pixels. 
            # We must reshape them into 28x28x1 images.
            # data = data.reshape(-1, self.img_size, self.img_size, 1).astype('float32')
            data = data.reshape(-1, self.img_size, self.img_size, 1)
            
            # Normalize pixel values from 0-255 down to 0.0-1.0
            # data /= 255.0
            
            # Create the labels (0, 1, 2, etc.)
            labels = np.full(data.shape[0], i)
            
            x_data.append(data)
            y_data.append(labels)
            
        # Combine all classes into one giant array
        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)
        
        print(f"\nTotal dataset size: {len(x_data)} images")
        
        # Shuffle the data
        indices = np.random.permutation(len(x_data))
        x_data = x_data[indices]
        y_data = y_data[indices]
        
        # Split into 80% Training, 20% Validation
        split = int(0.8 * len(x_data))
        x_train, x_val = x_data[:split], x_data[split:]
        y_train, y_val = y_data[:split], y_data[split:]
        
        return (x_train, y_train), (x_val, y_val)

    def build_model(self):
        """Build CNN"""
        print("\nBuilding model...")
        
        model = keras.Sequential([
            layers.Rescaling(1./255, input_shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'), #input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, epochs=10, batch_size=128):
        print("\n" + "=" * 70)
        print("Quick Draw .NPY Training")
        print("=" * 70)
        
        # Load and prep the data
        (x_train, y_train), (x_val, y_val) = self.load_data(max_samples_per_class=25000)
        
        # Convert to tf.data for faster GPU feeding
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        eval_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.build_model()
        self.model.summary()
        
        callbacks = [
            keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1),
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy', verbose=1),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor='val_loss', verbose=1)
        ]
        
        print("\n" + "=" * 70)
        print("Starting training...")
        print("=" * 70 + "\n")
        
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=eval_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nEvaluating on validation set...")
        eval_loss, eval_acc = self.model.evaluate(eval_dataset)
        print(f"\nFinal Validation Accuracy: {eval_acc*100:.2f}%")
        
        return history
    
    def save_model(self, path='quickdraw_model'):
        print(f"\nSaving model to {path}...")
        self.model.save(f'{path}.keras')
        with open(f'{path}_classes.json', 'w') as f:
            json.dump(self.classes, f, indent=2)
        print(f"✓ Model saved to {path}.keras")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    
    trainer = QuickDrawNumpyTrainer(data_dir=args.data_dir)
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    trainer.save_model('quickdraw_model')
    print("\n✅ Training Complete! Run 'python export.py' next.")

if __name__ == "__main__":
    main()