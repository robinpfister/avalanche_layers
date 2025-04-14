from abc import ABC, abstractmethod

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

    @abstractmethod
    def calculate(self, layers):
        pass