# internal imports
from layer import Layer
from layer_calculator_factory import LayerCalculator

# external imports
import numpy as np

class SituationRiskCalculator(LayerCalculator): # TODO
    def __init__(self):
        self.required_layers = [Layer.DANGER_EARLY, Layer.DANGER_LATE]
    
    def calculate(self, layers):
        pass
    
class TopoRiskStandardCalculator(LayerCalculator): # DONE
    def __init__(self):
        self.required_layers = [Layer.SHAPE, Layer.SLOPE]
    
    def calculate(self, layers):
        slope = layers.get(Layer.SLOPE)
        shape = layers.get(Layer.SHAPE)
        
        slope_probability = np.where(slope > 50, (55 - slope) / 5, 1 / (1 + np.exp(-0.8 * (slope - 34)))**(1/1.3))
        slope_probability = np.where(slope >= 55, 0, slope_probability)

        shape_probability = np.where(shape == 1, 22/153, 0)
        shape_probability = np.where(shape == 2, 119/153, shape_probability)
        shape_probability = np.where(shape == 3, 135/153, shape_probability)
        shape_probability = np.where(shape == 4, 8/153, shape_probability)
        shape_probability = np.where(shape == 5, 153/153, shape_probability)
        shape_probability = np.where(shape == 6, 57/153, shape_probability)
        shape_probability = np.where(shape == 7, 23/153, shape_probability)
        shape_probability = np.where(shape == 8, 32/153, shape_probability)
        shape_probability = np.where(shape == 9, 18/153, shape_probability)

        risk = ((2 * slope_probability) + shape_probability) / 3

        return risk
    
class TopoRiskAthmCalculator(LayerCalculator): # DONE
    def __init__(self):
        self.required_layers = [Layer.RELEVANT_SLOPE_AREA_PROPERTIES.AVERAGE_SLOPE, Layer.RELEVANT_SLOPE_AREA_PROPERTIES.SIZE]
    
    def calculate(self, layers):
        avg_slope = layers.get(Layer.RELEVANT_SLOPE_AREA_PROPERTIES.AVERAGE_SLOPE)
        size = layers.get(Layer.RELEVANT_SLOPE_AREA_PROPERTIES.SIZE)
        
        slope_probability = np.where(avg_slope > 50, (55 - avg_slope) / 5, 1 / (1 + np.exp(-0.8 * (avg_slope - 34)))**(1/1.3))
        slope_probability = np.where(avg_slope >= 55, 0, slope_probability)

        size_probability = np.where(size >= 400, 0.2, 0)
        size_probability = np.where(size >= 1600, 0.4, size_probability)
        size_probability = np.where(size >= 6400, 0.6, size_probability)
        size_probability = np.where(size >= 12800, 0.8, size_probability)
        size_probability = np.where(size >= 25600, 1, size_probability)

        risk = ((2 * slope_probability) + size_probability) / 3

        return risk

class TopoRiskLevelCalculator(LayerCalculator): # DONE
    def __init__(self):
        self.required_layers = [Layer.TOPO_RISK]
    
    def calculate(self, layers):
        risk = layers.get(Layer.TOPO_RISK)

        risk_level = np.where(risk > 0.5, 1, 0)
        risk_level = np.where(risk >= 0.75, 2, risk_level)

        return risk_level
    
class CombinedRiskCalculator(LayerCalculator): # TODO
    def __init__(self):
        self.required_layers = [Layer.TOPO_RISK, Layer.SITUATION_RISK]
    
    def calculate(self, layers):
        pass
