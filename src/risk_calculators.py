# internal imports
from layer import Layer
from layer_calculator_factory import LayerCalculator

# external imports
import numpy as np

class TopoRiskCalculator(LayerCalculator): # Risk without RSA
    def __init__(self):
        self.required_layers = [Layer.SLOPE, Layer.SHAPE]
    
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

        topo_risk = ((2 * slope_probability) + shape_probability) / 3

        #check for nan values in input
        topo_risk = np.where(np.isnan(shape), np.nan, topo_risk)

        return topo_risk
    
class TopoRiskRsaCalculator(LayerCalculator):
    def __init__(self):
        self.required_layers = [Layer.RSA_MAX_SLOPE, Layer.SLOPE, Layer.SHAPE]
    
    def calculate(self, layers):
        max_slope = layers.get(Layer.RSA_MAX_SLOPE)
        slope = layers.get(Layer.SLOPE)
        shape = layers.get(Layer.SHAPE)

        max_slope = np.where(max_slope == 0, slope, max_slope)
        
        slope_probability = np.where(max_slope > 50, (55 - max_slope) / 5, 1 / (1 + np.exp(-0.8 * (max_slope - 34)))**(1/1.3))
        slope_probability = np.where(max_slope >= 55, 0, slope_probability)

        shape_probability = np.where(shape == 1, 22/153, 0)
        shape_probability = np.where(shape == 2, 119/153, shape_probability)
        shape_probability = np.where(shape == 3, 135/153, shape_probability)
        shape_probability = np.where(shape == 4, 8/153, shape_probability)
        shape_probability = np.where(shape == 5, 153/153, shape_probability)
        shape_probability = np.where(shape == 6, 57/153, shape_probability)
        shape_probability = np.where(shape == 7, 23/153, shape_probability)
        shape_probability = np.where(shape == 8, 32/153, shape_probability)
        shape_probability = np.where(shape == 9, 18/153, shape_probability)

        topo_risk = ((2 * slope_probability) + shape_probability) / 3

        #check for nan values in input
        topo_risk = np.where(np.isnan(shape), np.nan, topo_risk)

        return topo_risk

class TopoRiskLevelCalculator(LayerCalculator):
    def __init__(self):
        self.required_layers = [Layer.TOPO_RISK]
    
    def calculate(self, layers):
        topo_risk = layers.get(Layer.TOPO_RISK)

        topo_risk_level = np.where(topo_risk > 0.5, 1, 0)
        topo_risk_level = np.where(topo_risk >= 0.75, 2, topo_risk_level)

        #check for nan values in input
        topo_risk_level = np.where(np.isnan(topo_risk), np.nan, topo_risk_level)

        return topo_risk_level
    
class CombinedRiskCalculator(LayerCalculator):
    def __init__(self):
        self.required_layers = [Layer.TOPO_RISK, Layer.SITUATION_RISK]
    
    def calculate(self, layers):
        topo_risk = layers.get(Layer.TOPO_RISK)
        situation_risk = layers.get(Layer.SITUATION_RISK)

        situation_risk_factor = np.where(situation_risk > 1, 1 + (3 * (situation_risk - 1)), situation_risk)
        situation_risk_factor = np.where(situation_risk > 2, 4 + (12 * (situation_risk - 2)), situation_risk_factor)
        situation_risk_factor = np.where(situation_risk > 3, 16 + (48 * (situation_risk - 3)), situation_risk_factor)
        situation_risk_factor = np.where(situation_risk > 4, 64, situation_risk_factor)

        combined_risk = topo_risk * situation_risk_factor

        #check for nan values in input
        combined_risk = np.where(np.isnan(topo_risk), np.nan, combined_risk)

        return combined_risk

class CombinedRiskLevelCalculator(LayerCalculator):
    def __init__(self):
        self.required_layers = [Layer.COMBINED_RISK]
    
    def calculate(self, layers):
        combined_risk = layers.get(Layer.COMBINED_RISK)

        combined_risk_level = np.where(combined_risk > 0.9, 1, 0)
        combined_risk_level = np.where(combined_risk > 3.7, 2, combined_risk_level)
        combined_risk_level = np.where(combined_risk > 9, 3, combined_risk_level)
        combined_risk_level = np.where(combined_risk > 32, 4, combined_risk_level)

        combined_risk_level[110,110] = 4

        #check for nan values in input
        combined_risk_level = np.where(np.isnan(combined_risk), np.nan, combined_risk_level)

        return combined_risk_level
