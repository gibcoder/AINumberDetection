import numpy as np


class NueralNetwork:
    def __init__(self, input_size=784,hidden_layers=[512,512],output_size=10):
        self.input_size= input_size
        self.hidden_layers=hidden_layers
        self.output_size= output_size
        self.weights=[]
        self.biases=[]
        #input to hidden layers network
        self.weights.append(0.01*np.random.randn(input_size,hidden_layers[0]))
        self.biases.append(np.zeros((1,hidden_layers[0])))

        #hidden layer network
        for i in range(len(hidden_layers)-1):
            self.weights.append(0.01*np.random.randn(hidden_layers[i],hidden_layers[i+1]))
            self.biases.append(np.zeros((1,hidden_layers[i+1])))
        #hidden layers network to output
        self.weights.append(0.01*np.random.randn(hidden_layers[len(hidden_layers)-1],output_size))
        self.biases.append(np.zeros((1,output_size)))
    def forward(self, inputs):
        layers=[inputs]

        for i in range(len(self.weights)):
            #dot product to
            layers.append(np.dot(layers[-1],self.weights[i])+self.biases[i])
        
        return layers[-1]
        

    

aaa=NueralNetwork()
inputs = np.random.randn(1, 784)  # Random input of shape (1, 784)
output = aaa.forward(inputs)
print("Output of the network:", output)
