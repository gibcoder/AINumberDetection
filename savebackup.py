import numpy as np
from tensorflow.keras.datasets import mnist
import pickle
from PIL import Image, ImageDraw, ImageFont


image = Image.new('L', (28, 28), color=255)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 20)  
draw.text((7, 4), '4', font=font, fill=0) 
test_input = list(image.getdata())


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers=[512, 512], output_size=10, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        
        #self.weights.append(
            #(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(2. / input_size)).astype(np.float32)
        #)
        #self.biases.append(np.zeros((1, hidden_layers[0]), dtype=np.float32))
        
        #for i in range(len(hidden_layers) - 1):
        #    self.weights.append(
        #        (np.random.randn(hidden_layers[i], hidden_layers[i + 1]) * np.sqrt(2. / hidden_layers[i])).astype(np.float32)
        #    )
        #    self.biases.append(np.zeros((1, hidden_layers[i + 1]), dtype=np.float32))
    
        #self.weights.append(
        #    (np.random.randn(hidden_layers[-1], output_size) * np.sqrt(2. / hidden_layers[-1])).astype(np.float32)
        #)
        #self.biases.append(np.zeros((1, output_size), dtype=np.float32))
    
    def forward(self, inputs):
        layers = [inputs]
        for i in range(len(self.weights) - 1):
            dp = np.dot(layers[-1], self.weights[i]) + self.biases[i]
            activation = np.maximum(0, dp) #ReLU activation
            layers.append(activation)
        dp = np.dot(layers[-1], self.weights[-1]) + self.biases[-1]
        layers.append(dp)#output layer
        return layers
    
    def softmax(self, values):
        exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
        exp_values_sum = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / exp_values_sum
    
    def cross_entropy_loss(self, output, correct_output):
        epsilon = 1e-9
        return -np.log(output[0, correct_output] + epsilon)
    
    def backpass(self, inputs, output, correct_output):

        #forward pass for layer activations
        layers = self.forward(inputs)
        grads_w = []
        grads_b = []
        expected_output = np.zeros_like(output)
        expected_output[0, correct_output] = 1
        error = output - expected_output  
        
        #grad for output
        dW = np.dot(layers[-2].T, error)
        dB = np.sum(error, axis=0, keepdims=True)
        grads_w.append(dW)
        grads_b.append(dB)
        
        #backpropogate through hidden
        for i in range(len(self.hidden_layers) -1, -1, -1):
            error = np.dot(error, self.weights[i +1].T)
            error[layers[i +1] <= 0] = 0  # Derivative of ReLU
            dW = np.dot(layers[i].T, error)
            dB = np.sum(error, axis=0, keepdims=True)
            grads_w.insert(0, dW)
            grads_b.insert(0, dB)
        #updates weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]
    
    def training(self, database_input, correct_output, epochs):

        for epoch in range(epochs):
            # shuffles for each epoch
            permutation = np.random.permutation(len(database_input))
            shuffled_input = database_input[permutation]
            shuffled_output = correct_output[permutation]
            
            total_loss = 0
            for DBinput, DBoutput in zip(shuffled_input, shuffled_output):
                DBinput = DBinput.reshape(1, -1)
                layers = self.forward(DBinput)
                output = self.softmax(layers[-1])
                loss = self.cross_entropy_loss(output, DBoutput)
                total_loss += loss
                self.backpass(DBinput, output, DBoutput)
            
            average_loss = total_loss / len(shuffled_input)
            print(f"Epoch {epoch +1}/{epochs}, Loss: {average_loss:.4f}")

def main():
    #load mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    #reshape and normalise
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255
    
    
    input_size = 784  
    nn = NeuralNetwork(input_size=input_size, hidden_layers=[512, 512], output_size=10, learning_rate=0.01)
    with open('nn_weights.pkl', 'rb') as f:
        loaded_weights, loaded_biases = pickle.load(f)
        nn.weights = loaded_weights
        nn.biases = loaded_biases
    
    ##print("training...")
    ##nn.training(x_train[:10000], y_train[:10000], epochs=50)  # Using 10,000 samples for faster training
    ##print("training completed.")
    ##with open('nn_biases_weights.pkl', 'wb') as f:
        ##pickle.dump((nn.weights, nn.biases), f)
    ##print("Weights and biases saved to 'nn_weights.pkl'.")
    

    ##alter the input from database here
    
    for i in range(100):
        test_input = x_test[i].reshape(1, -1)
        predicted_digit, output_probs = predict(nn, test_input)
    
    print("\n--- Test Result ---")
    print("Output probabilities:", output_probs)
    print("Predicted digit:", predicted_digit)
    print("Actual digit:", y_test[5])

def predict(nn, input_data):
    layers = nn.forward(input_data)
    output = nn.softmax(layers[-1])
    return np.argmax(output), output

if __name__ == "__main__":
    main()