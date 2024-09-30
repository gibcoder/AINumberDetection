import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pickle

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = Image.new('L', (28, 28), color=255)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 20)  
draw.text((7, 4), '4', font=font, fill=0) 
pixel_values = list(image.getdata())

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.learning_rate = 0.001


        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        for i in range(len(hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))

        
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):
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

    def score_system(self, output, correct_output):
        expected_output = np.zeros_like(output)
        expected_output[0, correct_output] = 1
        score_list = (output - expected_output) ** 2
        return np.sum(score_list)

    def backpass(self, inputs, output, correct_output):
        layers = self.forward(inputs)
        grads_w = []
        grads_b = []

        expected_output = np.zeros_like(output)
        expected_output[0, correct_output] = 1
        error = output - expected_output

        dW = np.dot(layers[-2].T, error)
        dB = np.sum(error, axis=0, keepdims=True)
        grads_w.append(dW)
        grads_b.append(dB)

        #backpropogate
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            error = np.dot(error, self.weights[i + 1].T)
            error[layers[i + 1] <= 0] = 0  

            dW = np.dot(layers[i].T, error)
            dB = np.sum(error, axis=0, keepdims=True)
            grads_w.insert(0, dW)
            grads_b.insert(0, dB)

 
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def training(self, database_input, correct_output, epochs):
        for iteration_number in range(epochs):
            for DBinput, DBoutput in zip(database_input, correct_output):
                DBinput = DBinput.reshape(1, -1)
                layers = self.forward(DBinput)
                output = self.softmax(layers[-1])
                self.backpass(DBinput, output, DBoutput)





x_train = x_train.reshape(-1, 28*28).astype('float32')
x_test = x_test.reshape(-1, 28*28).astype('float32')

#Normalize 0-1
x_train /= 255
x_test /= 255




input_size = 784  
nn = NeuralNetwork(input_size=input_size)
##with open('nn_weights.pkl', 'rb') as f:
    ##nn.weights, nn.biases = pickle.load(f)
nn.training(x_train[:1000], y_train[:1000], epochs=10)
with open('nn_weights.pkl', 'wb') as f:
    pickle.dump((nn.weights, nn.biases), f)
test_input = x_test[0].reshape(1, -1)
unrealised = nn.forward(test_input)
output = nn.softmax(unrealised[-1])

print("Output probabilities:", output)
print("Predicted digit:", np.argmax(output))