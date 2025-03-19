import numpy as np
from src.layers import Layers
from scipy.linalg import lstsq
from matplotlib import pyplot
from src.layer_calculator_factory import LayerCalculator
import multiprocessing as mp
from numba import jit, prange 
import numpy as np
import numba as nb
from numba.typed import List
import time
import matplotlib.pyplot as plt

import numpy as np

class CalculatorPreprocessing(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layers.HEIGHT]
        self.L = 1
        self.grid_size = 21

    def calculate(self, layers):
        heights = np.copy(layers.get(Layers.HEIGHT)).astype(np.float32)

        #TODO : Funktion zur Einstellung der Rasterbreite
        print(heights.shape)

        # QUANTITATIVE ANALYSIS OF LAND SURFACE TOPOGRAPHY, Zevenbergen S. 49 Formel (3) - (11)
        # coefficients, V, mask = self.get_polynomial_coeffs(heights, self.grid_size)
        coefficients = self.get_polynomial_coeffs(heights, self.grid_size)
        V = np.zeros((10000, 10000), dtype=np.float32)

        A, B, C = coefficients[..., 0], coefficients[..., 1], coefficients[..., 2]
        D, E, F = coefficients[..., 3], coefficients[..., 4], coefficients[..., 5]
        G, H, I = coefficients[..., 6], coefficients[..., 7], coefficients[..., 8]

        return np.array([A, B, C, D, E, F, G, H, I, V], dtype=np.float32)
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def get_polynomial_coeffs_adaptwnw(data, grid_width):
        yLength, xLength = data.shape[:2]  # Rastergröße
        coeffs = np.zeros((yLength, xLength, 9), dtype=np.float32)  # Speichert Koeffizienten [A, B, C, D, E, F, G, H, I] für jeden Datenpunkt
        mask = np.zeros((yLength, xLength), dtype=np.bool_)
        variability = np.zeros((yLength, xLength), dtype=np.float32)

        for i in range(10):
            full_grid = grid_width - int(2*i)
            half_grid = full_grid // 2  # Mitte des Fensters (10 Pixel Abstand)

            #-----------------------------DESIGN_MATRIX--------------------------------
            # Design-Matrix für das Polynom (21x21 = 441 Punkte, 9 Terme)
            X = np.zeros((full_grid**2, 9), dtype=np.float32)
            idx = 0
            for y in range(-half_grid, half_grid + 1):
                for x in range(-half_grid, half_grid + 1):
                    X[idx, 0] = x**2 * y**2  # A * X²Y²
                    X[idx, 1] = x**2 * y  # B * X²Y
                    X[idx, 2] = x * y**2  # C * XY²
                    X[idx, 3] = x**2  # D * X²
                    X[idx, 4] = y**2  # E * Y²
                    X[idx, 5] = x * y  # F * XY
                    X[idx, 6] = x  # G * X
                    X[idx, 7] = y  # H * Y
                    X[idx, 8] = 1  # I (Konstante)
                    idx += 1
            # Precompute (X^T @ X)^(-1) @ X^T
            XtX = np.dot(X.T, X)
            XtX_inv = np.linalg.inv(XtX)
            XtX_inv_Xt = np.dot(XtX_inv, X.T)  # (9x441) Matrix für LGS-Lösung
            #--------------------------------------------------------------------------
            #-------------------------XY-KOORDINATEN-RASTER----------------------------
            X_MESH = np.zeros((full_grid ** 2), dtype=np.float32)
            Y_MESH = np.zeros((full_grid ** 2), dtype=np.float32)
            idx = 0
            for y in range(-half_grid, half_grid + 1):
                for x in range(-half_grid, half_grid + 1):
                    Y_MESH[idx] = y
                    X_MESH[idx] = x
                    idx += 1           
            #--------------------------------------------------------------------------

            # Parallelisierte Berechnung für jede Zelle im Raster
            for y in nb.prange(half_grid, yLength - half_grid):
                for x in range(half_grid, xLength - half_grid):
                    if mask[y, x] == False:
                        Z = data[y - half_grid : y + half_grid + 1, x - half_grid : x + half_grid + 1].astype(np.float32)  # Z-Werte (441,)
                        Z_raveled = Z.ravel()
                        c = np.dot(XtX_inv_Xt, Z_raveled)
                        E = (c[0] * (X_MESH**2*Y_MESH**2) + 
                             c[1] * (X_MESH**2*Y_MESH) +
                             c[2] * (X_MESH*Y_MESH**2) + 
                             c[3] * (X_MESH**2) +
                             c[4] * (Y_MESH**2) +
                             c[5] * (X_MESH*Y_MESH) +
                             c[6] * (X_MESH) +
                             c[7] * (Y_MESH) +
                             c[8])
                        E = E.astype(np.float32)
                        Z_flat = Z.flatten()
                        diff = np.absolute(Z_flat - E)
                        v = np.average(diff)
                        if v < 1:
                            coeffs[y, x] = c
                            variability[y, x] = v
                            mask[y, x] = True
                        

        return coeffs, variability, mask  # (9980, 9980, 9) mit den Polynomkoeffizienten
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def get_polynomial_coeffs(data, grid_width):
        yLength, xLength = data.shape[:2]  # Rastergröße
        coeffs = np.zeros((yLength, xLength, 9), dtype=np.float32)  # Speichert Koeffizienten [A, B, C, D, E, F, G, H, I] für jeden Datenpunkt

        half_grid = grid_width // 2  # Mitte des Fensters (10 Pixel Abstand)

        # Design-Matrix für das Polynom (21x21 = 441 Punkte, 9 Terme)
        X = np.zeros((grid_width**2, 9), dtype=np.float32)
        idx = 0
        for y in range(-half_grid, half_grid + 1):
            for x in range(-half_grid, half_grid + 1):
                X[idx, 0] = x**2 * y**2  # A * X²Y²
                X[idx, 1] = x**2 * y  # B * X²Y
                X[idx, 2] = x * y**2  # C * XY²
                X[idx, 3] = x**2  # D * X²
                X[idx, 4] = y**2  # E * Y²
                X[idx, 5] = x * y  # F * XY
                X[idx, 6] = x  # G * X
                X[idx, 7] = y  # H * Y
                X[idx, 8] = 1  # I (Konstante)
                idx += 1

        # Precompute (X^T @ X)^(-1) @ X^T
        XtX = np.dot(X.T, X)
        XtX_inv = np.linalg.inv(XtX)
        XtX_inv_Xt = np.dot(XtX_inv, X.T)  # (9x441) Matrix für LGS-Lösung

        # Parallelisierte Berechnung für jede Zelle im Raster
        for y in nb.prange(half_grid, yLength - half_grid):
            for x in range(half_grid, xLength - half_grid):
                Z = data[y - half_grid : y + half_grid + 1, x - half_grid : x + half_grid + 1].ravel().astype(np.float32)  # Z-Werte (441,)
                coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  # Lösen des LGS

        return coeffs  # (9980, 9980, 9) mit den Polynomkoeffizienten

class CalculatorDirection(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layers.PREPROCESSED_DATA.G, Layers.PREPROCESSED_DATA.H]
        
    def calculate(self, layers):
        G = layers.get(Layers.PREPROCESSED_DATA.G)
        H = layers.get(Layers.PREPROCESSED_DATA.H)
            
        # Quelle : QUANTITATIVE ANALYSIS OF LAND SURFACE TOPOGRAPHY, Zevenbergen S. 50 Formel (15)
        # Degree :  0-360 (0-W, 90-N, 180-E, 270-S, 360-W)
        direction = np.arctan2(-H, -G) 
        direction = (direction * (180 / np.pi)) + 180

        return direction

class CalculatorDirectionLevel(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layers.DIRECTION]
    
    def calculate(self, layers):
        direction = layers.get(Layers.DIRECTION)

        direction_level = np.where(direction > 22.5, 2, 1)
        direction_level = np.where(direction > 67.5, 3, direction_level)
        direction_level = np.where(direction > 112.5, 4, direction_level)
        direction_level = np.where(direction > 157.5, 5, direction_level)
        direction_level = np.where(direction > 202.5, 6, direction_level)
        direction_level = np.where(direction > 247.5, 7, direction_level)
        direction_level = np.where(direction > 292.5, 8, direction_level)
        direction_level = np.where(direction > 337.5, 1, direction_level)
        
        return direction_level
        
class CalculatorSlope(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layers.PREPROCESSED_DATA.G, Layers.PREPROCESSED_DATA.H]
    
    def calculate(self, layers):
        G = layers.get(Layers.PREPROCESSED_DATA.G)
        H = layers.get(Layers.PREPROCESSED_DATA.H)
        
        # Quelle : QUANTITATIVE ANALYSIS OF LAND SURFACE TOPOGRAPHY, Zevenbergen S. 50 Formel (13)
        slope = ((G ** 2) + (H ** 2)) ** (1/2)
        slope = np.arctan((slope)) * (180 / np.pi)
        
        return slope

class CalculatorSlopeLevel(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layers.SLOPE]
    
    def calculate(self, layers):
        slope = layers.get(Layers.SLOPE)

        slope_level = np.where(slope > 20, 2, 1)
        slope_level = np.where(slope > 30, 3, slope_level)
        slope_level = np.where(slope > 40, 4, slope_level)
        slope_level = np.where(slope > 50, 5, slope_level)

        return slope_level

class CalculatorProfileCurvature(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layers.PREPROCESSED_DATA.D, Layers.PREPROCESSED_DATA.E, Layers.PREPROCESSED_DATA.F, Layers.PREPROCESSED_DATA.G, Layers.PREPROCESSED_DATA.H]

        self.z_factor = 1
    
    def calculate(self, layers):
        D = layers.get(Layers.PREPROCESSED_DATA.D)
        E = layers.get(Layers.PREPROCESSED_DATA.E)
        F = layers.get(Layers.PREPROCESSED_DATA.F)
        G = layers.get(Layers.PREPROCESSED_DATA.G)
        H = layers.get(Layers.PREPROCESSED_DATA.H)

        # Quelle :  QUANTITATIVE ANALYSIS OF LAND SURFACE TOPOGRAPHY, Zevenbergen S. 50 Formel (17) und Erklärung für Faktor 100
        #           z_factor - Curvature (3D Analyst) ArcGIS Pro, -+ Ausrichtung aus Text ableitbar
        denominator = (G ** 2) + (H ** 2)
        denominator = np.where(denominator == 0, np.nan, denominator)
        profile_curvature = ((2 * ((D * (G ** 2)) + (E * (H ** 2)) + (F * G * H))) / denominator) * (100 / (10 *self.z_factor))

        return profile_curvature

class CalculatorPlanCurvature(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layers.PREPROCESSED_DATA.D, Layers.PREPROCESSED_DATA.E, Layers.PREPROCESSED_DATA.F, Layers.PREPROCESSED_DATA.G, Layers.PREPROCESSED_DATA.H]

        self.z_factor = 1
    
    def calculate(self, layers):
        D = layers.get(Layers.PREPROCESSED_DATA.D)
        E = layers.get(Layers.PREPROCESSED_DATA.E)
        F = layers.get(Layers.PREPROCESSED_DATA.F)
        G = layers.get(Layers.PREPROCESSED_DATA.G)
        H = layers.get(Layers.PREPROCESSED_DATA.H)

        # Quelle :  QUANTITATIVE ANALYSIS OF LAND SURFACE TOPOGRAPHY, Zevenbergen S. 50 Formel (18) und Erklärung für Faktor 100
        #           z_factor - Curvature (3D Analyst) ArcGIS Pro, -+ Ausrichtung aus Text ableitbar
        denominator = (G ** 2) + (H ** 2)
        denominator = np.where(denominator == 0, np.nan, denominator)
        plan_curvature = ((-2 * ((D * (H ** 2)) + (E * (G ** 2)) - (F * G * H))) / denominator) * (100 / (10*self.z_factor))

        return plan_curvature

class CalculatorShape(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layers.PROFILE_CURVATURE, Layers.PLAN_CURVATURE]
    
    def calculate(self, layers):
        profile_curvature = np.copy(layers.get(Layers.PROFILE_CURVATURE))
        plan_curvature = np.copy(layers.get(Layers.PLAN_CURVATURE))

        # Quelle : Terrain analysis of skier-triggered avalanche starting zones, Vontobel S.372 Threshold_Values, Classification
        shape = np.where(profile_curvature < -0.2, 1, 2)
        shape = np.where(profile_curvature > 0.2, 3, shape)
        shape = np.where(plan_curvature < -0.2, shape, shape+3)
        shape = np.where(plan_curvature > 0.2, shape+3, shape)

        return shape

class CalculatorAvalancheRisk(LayerCalculator):
    def __init__(self):
        self.required_layers = [Layers.SHAPE, Layers.SLOPE_LEVEL]
    
    def calculate(self, layers):
        shape = np.copy(layers.get(Layers.SHAPE))
        slope = np.copy(layers.get(Layers.SLOPE_LEVEL))

        shape = np.piecewise(shape, [shape == 1, shape == 2, shape == 3, shape == 4, shape == 5, shape == 6, shape == 7, shape == 8, shape == 9], [0.125, 0.75, 0.75, 0.0625, 1, 0.375, 0.125, 0.25, 0.125])
        slope = np.piecewise(slope, [slope == 1, slope == 2, slope == 3, slope == 4, slope == 5], [0, 0.2, 1, 0.2, 0])

        avalanche_risk = shape * slope

        return avalanche_risk