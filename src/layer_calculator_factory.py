from abc import ABC, abstractmethod
import numpy as np

class Size():
    def __init__(self, x : int, y : int):
        self.x = x
        self.y = y

class LayerCalculatorFactory():

    def __init__(self):
        self.calculators = {}

    def register_calculator(self, layer_type, calculator):
        self.calculators[layer_type] = calculator

    def get_calculator(self, layer_type):
        return self.calculators.get(layer_type)()
    
class LayerCalculator(ABC):

    def __init__(self):
        self.required_layers = []
    
    def getSize(self, layers):
        layersData = layers.values()
        x = []
        y = []
        for layer in layersData:
            x.append(np.size(layer, 1))
            y.append(np.size(layer, 0))
        if x.count(x[0]) == len(x) and y.count(y[0]) == len(y):     #Check if all layers have the same length
            self.xLength = x[0]
            self.yLength = y [0]
            return Size(x[0], y[0])
        return None                                                 #Error

    @abstractmethod
    def calculate(self, layers):
        pass