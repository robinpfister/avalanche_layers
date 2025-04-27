# internal imports
from layer import Layer
from layer_calculator_factory import LayerCalculator

# external imports
import numpy as np
import numba as nb

class EvansCoefficientCalculator(LayerCalculator):
    def __init__(self):
        self.required_layers = [Layer.HEIGHT]
        self.L = 1
        self.grid_size = 21

    def calculate(self, layers):
        heights = np.copy(layers.get(Layer.HEIGHT))

        coefficients = self.calculate_coefficients(heights, self.grid_size)

        A, B, C = coefficients[..., 0], coefficients[..., 1], coefficients[..., 2]
        D, E, F = coefficients[..., 3], coefficients[..., 4], coefficients[..., 5]

        # existing because of ZevenbergenCoefficientCalculator -> same interface
        G = np.zeros(np.shape(heights), dtype=np.float32)
        H = np.zeros(np.shape(heights), dtype=np.float32)
        I = np.zeros(np.shape(heights), dtype=np.float32)

        return np.array([A, B, C, D, E, F, G, H, I])
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def calculate_coefficients(data, grid_width):
        
        data = data.astype(np.float32)
        yLength, xLength = data.shape
        coefficients = np.zeros((yLength, xLength, 6), dtype=np.float32)
        half_grid = int((grid_width-1) / 2)

        # F-Matrix
        X = np.zeros((grid_width*grid_width, 6), dtype=np.float32)
        i = 0
        for y in range(-half_grid, half_grid + 1):
            for x in range(-half_grid, half_grid + 1):
                X[i, 0] = x**2    # A
                X[i, 1] = y**2    # B
                X[i, 2] = x*y     # C 
                X[i, 3] = x       # D
                X[i, 4] = y       # E
                X[i, 5] = 1       # F
                i += 1

        # calculate least-squares-formular
        XtX = np.dot(X.T, X)
        XtX_inv = np.linalg.inv(XtX)
        XtX_inv_Xt = np.dot(XtX_inv, X.T)

        # calculate coefficients for every data point
        for y in nb.prange(half_grid, yLength - half_grid):
            for x in range(half_grid, xLength - half_grid):
                Z = data[y - half_grid : y + half_grid + 1, x - half_grid : x + half_grid + 1].ravel().astype(np.float32)
                coefficients[y, x] = np.dot(XtX_inv_Xt, Z)

        return coefficients
    
class ZevenbergenCoefficientCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.HEIGHT]
        self.L = 1
        self.grid_size = 21

    def calculate(self, layers):
        heights = np.copy(layers.get(Layer.HEIGHT)).astype(np.float32)

        coefficients = self.get_polynomial_coeffs(heights, self.grid_size)

        A, B, C = coefficients[..., 0], coefficients[..., 1], coefficients[..., 2]
        D, E, F = coefficients[..., 3], coefficients[..., 4], coefficients[..., 5]
        G, H, I = coefficients[..., 6], coefficients[..., 7], coefficients[..., 8]

        return np.array([G, H, I, A, B, C, D, E, F], dtype=np.float32)
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def get_polynomial_coeffs(data, grid_width):
        yLength, xLength = data.shape[:2]
        coeffs = np.zeros((yLength, xLength, 9), dtype=np.float32)

        half_grid = grid_width // 2

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
                X[idx, 8] = 1  # I 
                idx += 1

        XtX = np.dot(X.T, X)
        XtX_inv = np.linalg.inv(XtX)
        XtX_inv_Xt = np.dot(XtX_inv, X.T)

        # Parallelisierte Berechnung für jede Zelle im Raster
        for y in nb.prange(half_grid, yLength - half_grid):
            for x in range(half_grid, xLength - half_grid):
                Z = data[y - half_grid : y + half_grid + 1, x - half_grid : x + half_grid + 1].ravel().astype(np.float32)  # Z-Werte (441,)
                coeffs[y, x] = np.dot(XtX_inv_Xt, Z)

        return coeffs

class AdaptiveWindowCoefficientCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.HEIGHT]
        self.L = 1
        self.grid_size = 21

    def calculate(self, layers):
        heights = np.copy(layers.get(Layer.HEIGHT)).astype(np.float32)

        coefficients = self.get_polynomial_coeffs_adaptwnw(heights, self.grid_size)

        A, B, C = coefficients[..., 0], coefficients[..., 1], coefficients[..., 2]
        D, E, F = coefficients[..., 3], coefficients[..., 4], coefficients[..., 5]
        G, H, I = coefficients[..., 6], coefficients[..., 7], coefficients[..., 8]

        return np.array([G, H, I, A, B, C, D, E, F], dtype=np.float32)
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def get_polynomial_coeffs_adaptwnw(data, grid_width):
        yLength, xLength = data.shape[:2]
        coeffs = np.zeros((yLength, xLength, 9), dtype=np.float32)
        mask = np.zeros((yLength, xLength), dtype=np.bool_)
        variability = np.zeros((yLength, xLength), dtype=np.float32)

        for i in range(10):
            full_grid = grid_width - int(2*i)
            half_grid = full_grid // 2

            #-----------------------------DESIGN_MATRIX--------------------------------
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
                    X[idx, 8] = 1  # I
                    idx += 1
            XtX = np.dot(X.T, X)
            XtX_inv = np.linalg.inv(XtX)
            XtX_inv_Xt = np.dot(XtX_inv, X.T)
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
                        Z = data[y - half_grid : y + half_grid + 1, x - half_grid : x + half_grid + 1].astype(np.float32)
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
                        
        return coeffs
