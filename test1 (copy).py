# import tensorflow as tf
# from tensorflow.keras import layers

# model = tf.keras.Sequential()
# # Adds a densely-connected layer with 64 units to the model:
# model.add(layers.Dense(64, activation='relu'))
# # Add another:
# model.add(layers.Dense(64, activation='relu'))
# # Add a softmax layer with 10 output units:
# model.add(layers.Dense(10, activation='softmax'))



# # Create a sigmoid layer:
# layers.Dense(64, activation='sigmoid')
# # Or:
# layers.Dense(64, activation=tf.sigmoid)

# # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
# layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
# layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# # A linear layer with a kernel initialized to a random orthogonal matrix:
# layers.Dense(64, kernel_initializer='orthogonal')

# # A linear layer with a bias vector initialized to 2.0s:
# layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))



# model = tf.keras.Sequential([
# # Adds a densely-connected layer with 64 units to the model:
# layers.Dense(64, activation='relu'),
# # Add another:
# layers.Dense(64, activation='relu'),
# # Add a softmax layer with 10 output units:
# layers.Dense(10, activation='softmax')])

# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])



# # Configure a model for mean-squared error regression.
# model.compile(optimizer=tf.train.AdamOptimizer(0.01),
#               loss='mse',       # mean squared error
#               metrics=['mae'])  # mean absolute error

# # Configure a model for categorical classification.
# model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#               loss=tf.keras.losses.categorical_crossentropy,
#               metrics=[tf.keras.metrics.categorical_accuracy])



# import numpy as np

# data = np.random.random((1000, 32))
# labels = np.random.random((1000, 10))

# model.fit(data, labels, epochs=10, batch_size=32)




# import numpy as np

# data = np.random.random((1000, 32))
# labels = np.random.random((1000, 10))

# val_data = np.random.random((100, 32))
# val_labels = np.random.random((100, 10))

# model.fit(data, labels, epochs=10, batch_size=32,
#           validation_data=(val_data, val_labels))





# # Instantiates a toy dataset instance:
# dataset = tf.data.Dataset.from_tensor_slices((data, labels))
# dataset = dataset.batch(32)
# dataset = dataset.repeat()

# # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
# model.fit(dataset, epochs=10, steps_per_epoch=30)





# dataset = tf.data.Dataset.from_tensor_slices((data, labels))
# dataset = dataset.batch(32).repeat()

# val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
# val_dataset = val_dataset.batch(32).repeat()

# model.fit(dataset, epochs=10, steps_per_epoch=30,
#           validation_data=val_dataset,
#           validation_steps=3)





# data = np.random.random((1000, 32))
# labels = np.random.random((1000, 10))

# model.evaluate(data, labels, batch_size=32)

# model.evaluate(dataset, steps=30)





# result = model.predict(data, batch_size=32)
# print(result.shape)




import tensorflow as tf
mnist = tf.keras.datasets.mnist

#train - training set. x- input, y - output.
#test- error calc. evaluate in the end 
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#pixel intensity. make equal 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0


#configure model. 4 layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  #1st param num of output
  #activation - result normalization
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#inner config
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#launch training
model.fit(x_train, y_train, epochs=5)

#check accuracy
#separation of concerns
model.evaluate(x_test, y_test)