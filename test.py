import numpy as np
import os
current_dir = os.getcwd()

import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Training data shape: {x_train.shape}")

image = Image.new('L', (28, 28), color=255)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 20)  
draw.text((7, 4), '1', font=font, fill=0) 
#image.show()
pixel_values = list(image.getdata())##will become database input from the pixelvalues just a temporary to test


class NueralNetwork:
    def __init__(self, input_size, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.learning_rate = 0.001

        #Input to hidden layers
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))

        # Hidden to output layer
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):#forward function
        layers = [inputs]
        for i in range(len(self.weights) - 1):
            dp = np.dot(layers[-1], self.weights[i]) + self.biases[i]
            activation = np.maximum(0, dp)
            layers.append(activation)
        dp = np.dot(layers[-1], self.weights[-1]) + self.biases[-1]
        layers.append(dp)
        return layers

    def softmax(self, values):
        exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
        exp_values_sum = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / exp_values_sum

    def scoreSystem(self, output, CorrectOutput):
        scoreList = []
        for i in range(len(output[0])):
            expectedChance = 1 if i == CorrectOutput else 0
            scoreList.append((output[0, i] - expectedChance) ** 2)
        added1 = sum(scoreList)

        return added1

    def backpass(self, inputs, output, CorrectOutput):
        layers = self.forward(inputs)
        grads_w = []
        grads_b = []

        # Compute gradient for output layer
        expected_output = np.zeros_like(output)
        expected_output[0, CorrectOutput] = 1
        error = output - expected_output

        dW = np.dot(layers[-2].T, error)
        dB = np.sum(error, axis=0, keepdims=True)
        grads_w.append(dW)
        grads_b.append(dB)

        # Backpropagate through hidden layers
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            error = np.dot(error, self.weights[i + 1].T)
            error[layers[i + 1] <= 0] = 0  # ReLU derivative

            dW = np.dot(layers[i].T, error)
            dB = np.sum(error, axis=0, keepdims=True)
            grads_w.insert(0, dW)
            grads_b.insert(0, dB)

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def training(self, databaseInput, CorrectOutput, numberOfIterations):
        for iterationNumber in range(numberOfIterations):
            for DBinput, DBoutput in zip(databaseInput, CorrectOutput):
                DBinput = DBinput.reshape(1, -1)
                layers = self.forward(DBinput)
                output = self.softmax(layers[-1])
                self.backpass(DBinput, output, DBoutput)
    
# Example usage
input_size = 784  
nn = NueralNetwork(input_size=input_size)

# Example input
inputs = np.array(pixel_values).reshape(1, -1)
unrealised = nn.forward(inputs)
output = nn.softmax(unrealised[-1])

# Assuming CorrectOutput is 1
CorrectOutput = 1
added = nn.scoreSystem(output, CorrectOutput)
nn.backpass(inputs, output, CorrectOutput)


nn.training(inputs, CorrectOutput, 1)
print("Output probabilities:", output)
print("Loss:", added)
print("Sum of probabilities:", np.sum(output))