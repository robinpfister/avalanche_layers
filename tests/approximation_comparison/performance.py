import numpy as np
import time
from osgeo import gdal
from tabulate import tabulate
from numba import jit, prange, objmode
import numba as nb

def calc_second_order_least_squares(heights, window_size, grid_size):

    ################################################################################################################
    F = np.zeros((window_size * window_size, 6))
    coeffs = np.zeros((grid_size, grid_size, 6))
    idx = 0

    for y in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
        for x in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
            F[idx, 0] = x**2  # A
            F[idx, 1] = y**2  # B
            F[idx, 2] = x*y  # C 
            F[idx, 3] = x  # D
            F[idx, 4] = y  # E
            F[idx, 5] = 1  # F 
            idx += 1

    XtX = np.dot(F.T, F)
    XtX_inv = np.linalg.inv(XtX)
    XtX_inv_Xt = np.dot(XtX_inv, F.T)
    
    for y in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
        for x in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
            Z = heights[y - int((window_size-1)/2) : y + int((window_size-1)/2) + 1, 
                        x - int((window_size-1)/2) : x + int((window_size-1)/2) + 1].ravel().astype(np.float32)
            coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  
    
    #A, B, C, D, E, F = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3], coeffs[:, :, 4], coeffs[:, :, 5]
    ################################################################################################################
    
    ################################################################################################################
    #slope = ((D ** 2) + (E ** 2)) ** (1/2)
    ################################################################################################################

    ################################################################################################################
    #aspect = np.arctan2(-E, -D)
    ################################################################################################################

    ################################################################################################################
    #profilec = ((-2 * ((A * (D ** 2)) + (B * (E ** 2)) + (C * D * E))) / (((D ** 2) + (E ** 2)) * ((1 + (D ** 2) + (E ** 2)) ** (3/2))))
    ################################################################################################################

    ################################################################################################################
    #planc = ((-2 * ((B * (D ** 2)) + (A * (E ** 2)) - (C * D * E))) / (((D ** 2) + (E ** 2)) ** (3/2)))
    ################################################################################################################

    return None

def calc_fourth_order_least_squares(heights, window_size, grid_size):
    
    ################################################################################################################
    F = np.zeros((window_size * window_size, 9))
    coeffs = np.zeros((grid_size, grid_size, 9))
    idx = 0

    for y in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
        for x in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
            F[idx, 0] = x**2 * y**2  # D
            F[idx, 1] = x**2 * y  # E
            F[idx, 2] = x * y**2  # F 
            F[idx, 3] = x**2  # A
            F[idx, 4] = y**2  # B
            F[idx, 5] = x*y  # C 
            F[idx, 6] = x  # D
            F[idx, 7] = y  # E
            F[idx, 8] = 1  # F 
            idx += 1

    XtX = np.dot(F.T, F)
    XtX_inv = np.linalg.inv(XtX)
    XtX_inv_Xt = np.dot(XtX_inv, F.T)
    
    for y in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
        for x in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
            Z = heights[y - int((window_size-1)/2) : y + int((window_size-1)/2) + 1, 
                        x - int((window_size-1)/2) : x + int((window_size-1)/2) + 1].ravel().astype(np.float32)
            coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  
    
    #A, B, C, D, E, F, G, H, I = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3], coeffs[:, :, 4], coeffs[:, :, 5], coeffs[:, :, 6], coeffs[:, :, 7], coeffs[:, :, 8] 
    ################################################################################################################

    ################################################################################################################
    #slope = ((G ** 2) + (H ** 2)) ** (1/2)
    ################################################################################################################

    ################################################################################################################
    #aspect = np.arctan2(-H, -G)
    ################################################################################################################

    ################################################################################################################
    #profilec = ((-2 * ((D * (G ** 2)) + (E * (H ** 2)) + (F * G * H))) / (((G ** 2) + (H ** 2)) * ((1 + (G ** 2) + (H ** 2)) ** (3/2))))
    ################################################################################################################

    ################################################################################################################
    #planc = ((-2 * ((D * (H ** 2)) + (E * (G ** 2)) - (F * G * H))) / (((G ** 2) + (H ** 2)) ** (3/2)))
    ################################################################################################################

    return None

@nb.njit(parallel=True, fastmath=True)
def calc_second_order_least_squares_parallel(heights, window_size, grid_size):

    ################################################################################################################
    F = np.zeros((window_size * window_size, 6))
    coeffs = np.zeros((grid_size, grid_size, 6))
    idx = 0

    for y in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
        for x in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
            F[idx, 0] = x**2  # A
            F[idx, 1] = y**2  # B
            F[idx, 2] = x*y  # C 
            F[idx, 3] = x  # D
            F[idx, 4] = y  # E
            F[idx, 5] = 1  # F 
            idx += 1
    XtX = np.dot(F.T, F)
    XtX_inv = np.linalg.inv(XtX)
    XtX_inv_Xt = np.dot(XtX_inv, F.T).astype(np.float32)
    
    for y in prange(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
        for x in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
            Z = heights[y - int((window_size-1)/2) : y + int((window_size-1)/2) + 1, 
                        x - int((window_size-1)/2) : x + int((window_size-1)/2) + 1].ravel().astype(np.float32)
            coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  
    
    #A, B, C, D, E, F = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3], coeffs[:, :, 4], coeffs[:, :, 5]
    ################################################################################################################

    return None

@nb.njit(parallel=True, fastmath=True)
def calc_fourth_order_least_squares_parallel(heights, window_size, grid_size):
    
    ################################################################################################################
    F = np.zeros((window_size * window_size, 9))
    coeffs = np.zeros((grid_size, grid_size, 9))
    idx = 0

    for y in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
        for x in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
            F[idx, 0] = x**2 * y**2  # D
            F[idx, 1] = x**2 * y  # E
            F[idx, 2] = x * y**2  # F 
            F[idx, 3] = x**2  # A
            F[idx, 4] = y**2  # B
            F[idx, 5] = x*y  # C 
            F[idx, 6] = x  # D
            F[idx, 7] = y  # E
            F[idx, 8] = 1  # F 
            idx += 1

    XtX = np.dot(F.T, F)
    XtX_inv = np.linalg.inv(XtX)
    XtX_inv_Xt = np.dot(XtX_inv, F.T).astype(np.float32)
    
    for y in prange(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
        for x in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
            Z = heights[y - int((window_size-1)/2) : y + int((window_size-1)/2) + 1, 
                        x - int((window_size-1)/2) : x + int((window_size-1)/2) + 1].ravel().astype(np.float32)
            coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  
    
    #A, B, C, D, E, F, G, H, I = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3], coeffs[:, :, 4], coeffs[:, :, 5], coeffs[:, :, 6], coeffs[:, :, 7], coeffs[:, :, 8] 
    ################################################################################################################

    return None

def get_real_data(rows, cols):
    file = gdal.Open(f'data_layer/data_deutschland/height.tif')
    data = file.ReadAsArray()[0 : rows, 0 : cols]
    return data

def measure_time(func, arg1, arg2, arg3, samples):
    elapsed_time = np.full((samples), np.nan, dtype=np.float64)
    for sample in range(samples):
        start_time = time.time()#perf_counter()
        func(arg1, arg2, arg3)
        end_time = time.time()#perf_counter()
        elapsed_time[sample] = end_time - start_time
    print(elapsed_time)
    return np.average(elapsed_time * 1000)

def print_table(data, headers):
    print(tabulate(data, headers=headers, tablefmt="grid", floatfmt=".2f"))


data_size_list = [10, 100, 1000]
window_size_list = [3, 11, 21]
samples = 10

results = np.full((4, len(window_size_list), 3), np.nan)

#call numba-function once to remove overhead
data = get_real_data(10, 10)
calc_fourth_order_least_squares_parallel(data, 3, 10)
calc_second_order_least_squares_parallel(data, 3, 10)

for index0, window_size in enumerate(window_size_list):
    for index1, data_size in enumerate(data_size_list):

        adjusted_data_size = int((window_size - 1) / 2) + data_size
        data = get_real_data(adjusted_data_size, adjusted_data_size)

        results[0, index0, index1] = measure_time(calc_second_order_least_squares, data, window_size, adjusted_data_size, samples)
        results[1, index0, index1] = measure_time(calc_second_order_least_squares_parallel, data, window_size, adjusted_data_size, samples)
        results[2, index0, index1] = measure_time(calc_fourth_order_least_squares, data, window_size, adjusted_data_size, samples)
        results[3, index0, index1] = measure_time(calc_fourth_order_least_squares_parallel, data, window_size, adjusted_data_size, samples)

        print(f'{round(100 * (index0*len(data_size_list) + index1+1) / (len(window_size_list)*len(data_size_list)))}%')

headers = ['', '10x10', '100x100', '1000x1000']
data = [
    ["3x3", results[0, 0, 0], results[0, 0, 1], results[0, 0, 2]],
    ["11x11", results[0, 1, 0], results[0, 1, 1], results[0, 1, 2]],
    ["21x21", results[0, 2, 0], results[0, 2, 1], results[0, 2, 2]]]
print('Second Order')
print_table(data, headers)

data = [
    ["3x3", results[1, 0, 0], results[1, 0, 1], results[1, 0, 2]],
    ["11x11", results[1, 1, 0], results[1, 1, 1], results[1, 1, 2]],
    ["21x21", results[1, 2, 0], results[1, 2, 1], results[1, 2, 2]]]

print('Second Order Parallel')
print_table(data, headers)

data = [
    ["3x3", results[2, 0, 0], results[2, 0, 1], results[2, 0, 2]],
    ["11x11", results[2, 1, 0], results[2, 1, 1], results[2, 1, 2]],
    ["21x21", results[2, 2, 0], results[2, 2, 1], results[2, 2, 2]]]

print('Fourth Order')
print_table(data, headers)

data = [
    ["3x3", results[3, 0, 0], results[3, 0, 1], results[3, 0, 2]],
    ["11x11", results[3, 1, 0], results[3, 1, 1], results[3, 1, 2]],
    ["21x21", results[3, 2, 0], results[3, 2, 1], results[3, 2, 2]]]

print('Fourth Order Parallel')
print_table(data, headers)