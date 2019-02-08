import tensorflow as tf
from keras.optimizers import SGD
from metrics import precision, recall, f1

def build_model_mnist(input_shape=(28, 28, 1)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
    	optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', precision, recall, f1]
        )
    return model

def build_model_fashion_mnist(input_shape=(28, 28, 1)):
    model = tf.keras.models.Sequential([
    	tf.keras.layers.Flatten(input_shape=input_shape),

        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
    	optimizer="adam",
    	loss="sparse_categorical_crossentropy",
    	metrics=['accuracy', precision, recall, f1]
    	)
    return model


def build_model_pets(input_shape=(28, 28)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv2D(32, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.6),
        
        tf.keras.layers.Conv2D(64, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv2D(64, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1, activation="sigmoid")
                               ])
    model.compile(
    	optimizer="rmsprop",
		loss="binary_crossentropy",
		metrics=['accuracy', precision, recall, f1]
		)
    return model


def build_model_cifar10(input_shape=(32, 32, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv2D(32, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(64, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv2D(64, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(10, activation="softmax")
	])

	# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(
    	optimizer="adam",
        loss="sparse_categorical_crossentropy",
		metrics=['accuracy', precision, recall, f1]
		)
    return model


def build_model_cifar100(input_shape=(32, 32, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3,3), padding="same", input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv2D(128, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(256, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv2D(256, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(512, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv2D(512, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(1024),
		tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(100, activation="softmax")
    ])
    model.compile(
    	optimizer="adam",
		loss="sparse_categorical_crossentropy",
		metrics=['accuracy', precision, recall, f1]
		)
    return model
