# CSE4334-5334-Assignment-3-solution

Download Here: [CSE4334/5334 Assignment 3 solution](https://jarviscodinghub.com/assignment/cse4334-5334-assignment-3-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

Problem 1
(Tensorflow and Keras, 50pts) Try out the tutorial for Deep Learning using Tensorflow at https://www.
tensorflow.org/tutorials. It has the following lines of code.
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation=tf.nn.relu),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=’adam’,loss=’sparse_categorical_crossentropy’, metrics=[’accuracy’])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
1. (20pt) In the report, write comments for each line of code given above and explain what this framework
is doing.
2. (20pt) Change the number of hidden nodes to 10 and train this neural network. The trained model
contains the weights that it has learned from training. Plot the features in the hidden layer that it has
learned from training and include them in the report. That is, reshape the learned weights (vectors) in
Instructor: W. H. Kim (won.kim@uta.edu), TA: Priyank Arora (priyank.arora@mavs.uta.edu) Page 1 of 2
CSE4334/5334 Data Mining Assignment 3
the first layer (between the input and the 1st hidden layer) to the image dimension (in 2D) and show
them. (You will get 0 marks for this if the result is not included in the report.)
3. (10pt) Change the number of hidden nodes to 1, 10, 50 and 100 and report how the testing accuracy
changes for the testing dataset. Report the result and your observation in the report.

