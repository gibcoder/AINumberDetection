import numpy as np
class Layer():

    def init(self ,inputs=None,outputs=None):

        self.inputs=inputs
        self.outputs=outputs
        self.weights=[inputs,outputs]
        self.bias=[outputs]

    def layer(self,inputs,outputs):
        self.inputs=inputs
        self.outputs=outputs

'''class Nuerone:
    def __init__(self, bias, weights, inputs):
        
        self.weights=weights
        self.bias=bias
        self.inputs=inputs
        self.output=0

    def forward(self):
        return inputs


singularNuerone=Nuerone(
    bias=1.5,
    weights=[0.3,0.4,0.7,0.5]
    )'''



    





