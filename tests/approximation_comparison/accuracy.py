import numpy as np
import time
from osgeo import gdal
from tabulate import tabulate
from numba import jit, prange, objmode
import numba as nb

def calc_differences_properties(heights):
    dy, dx = np.gradient(heights, 1)
    dyy, _ = np.gradient(dy, 1)
    dxy, dxx = np.gradient(dx, 1)

    slope = np.sqrt((dy**2) + (dx**2))
    slope = np.arctan((slope)) * (180 / np.pi)

    aspect = np.arctan2(dy, dx)
    aspect = (aspect * (180 / np.pi)) + 180

    profilec = 100 * (-1 * ((dxx * (dx ** 2)) + (dyy * (dy ** 2)) + (2 * dx * dy * dxy))) / (((dx ** 2) + (dy ** 2)) * ((1 + (dx ** 2) + (dy ** 2)) ** (3/2)))

    planc = 100 * (-1 * ((dxx * (dy ** 2)) + (dyy * (dx ** 2)) - (2 * dy * dx * dxy))) / (((dx ** 2) + (dy ** 2)) ** (3/2))

    return slope, aspect, profilec, planc

def calc_second_order_properties(heights, window_size, grid_size):

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
    
    for y in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
        for x in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
            Z = heights[y - int((window_size-1)/2) : y + int((window_size-1)/2) + 1, 
                        x - int((window_size-1)/2) : x + int((window_size-1)/2) + 1].ravel().astype(np.float32)
            coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  
    
    A, B, C, D, E, F = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3], coeffs[:, :, 4], coeffs[:, :, 5]

    slope = ((D ** 2) + (E ** 2)) ** (1/2)
    slope = np.arctan((slope)) * (180 / np.pi)

    aspect = np.arctan2(E, D)
    aspect = (aspect * (180 / np.pi)) + 180

    profilec = 100 * (-2 * ((A * (D ** 2)) + (B * (E ** 2)) + (C * D * E))) / (((D ** 2) + (E ** 2)) * ((1 + (D ** 2) + (E ** 2)) ** (3/2)))

    planc = 100 * (-2 * ((B * (D ** 2)) + (A * (E ** 2)) - (C * D * E))) / (((D ** 2) + (E ** 2)) ** (3/2))

    return slope, aspect, profilec, planc

def calc_fourth_order_properties(heights, window_size, grid_size):
    
    ################################################################################################################
    F = np.zeros((window_size * window_size, 9))
    coeffs = np.zeros((grid_size, grid_size, 9))
    idx = 0

    for y in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
        for x in range(int(-(window_size-1)/2), int((window_size-1)/2 + 1)):
            F[idx, 0] = x**2 * y**2  # A
            F[idx, 1] = x**2 * y  # B
            F[idx, 2] = x * y**2  # C 
            F[idx, 3] = x**2  # D
            F[idx, 4] = y**2  # E
            F[idx, 5] = x*y  # F 
            F[idx, 6] = x  # G
            F[idx, 7] = y  # H
            F[idx, 8] = 1  # I 
            idx += 1

    XtX = np.dot(F.T, F)
    XtX_inv = np.linalg.inv(XtX)
    XtX_inv_Xt = np.dot(XtX_inv, F.T).astype(np.float32)
    
    for y in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
        for x in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
            Z = heights[y - int((window_size-1)/2) : y + int((window_size-1)/2) + 1, 
                        x - int((window_size-1)/2) : x + int((window_size-1)/2) + 1].ravel().astype(np.float32)
            coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  
    
    A, B, C, D, E, F, G, H, I = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3], coeffs[:, :, 4], coeffs[:, :, 5], coeffs[:, :, 6], coeffs[:, :, 7], coeffs[:, :, 8] 

    slope = ((G ** 2) + (H ** 2)) ** (1/2)
    slope = np.arctan((slope)) * (180 / np.pi)

    aspect = np.arctan2(H, G)
    aspect = (aspect * (180 / np.pi)) + 180

    profilec = 100 * (-2 * ((D * (G ** 2)) + (E * (H ** 2)) + (F * G * H))) / (((G ** 2) + (H ** 2)) * ((1 + (G ** 2) + (H ** 2)) ** (3/2)))

    planc = 100 * (-2 * ((D * (H ** 2)) + (E * (G ** 2)) - (F * G * H))) / (((G ** 2) + (H ** 2)) ** (3/2))

    return slope, aspect, profilec, planc

############################################################################

def get_real_data(length_y, length_x):
    file = gdal.Open(f'data_layer/data_deutschland/height.tif')
    data = file.ReadAsArray()[2000 : 2000 + length_y, 2000 : 2000 + length_x]
    return data

def get_center(data, window_size):
    data_size_y, data_size_x = data.shape
    window = data[int((data_size_y - window_size) / 2): int(data_size_y - ((data_size_y - window_size) / 2)), int((data_size_x - window_size) / 2): int(data_size_x - ((data_size_x - window_size) / 2))]

    return window

def print_table(data):
    print(tabulate(data, tablefmt="grid", floatfmt=".2f"))

def show_slope(data, window_size_list, show_window=3):
    slope_d, _, _, _ = calc_differences_properties(data)
    print('------------\nSlope\n------------')
    print_table(get_center(slope_d, window_size_list[-1] + 2))

    for window_size in window_size_list:

        slope_2, _, _, _ = calc_second_order_properties(data, window_size, data_size)
        slope_4, _, _, _ = calc_fourth_order_properties(data, window_size, data_size)

        print(f'\nWindow Size {window_size}\n')
        print('Differences')
        print_table(get_center(slope_d, show_window))
        print('Second Order')
        print_table(get_center(slope_2, show_window))
        print('Fourth Order')
        print_table(get_center(slope_4, show_window))

def show_aspect(data, window_size_list, show_window=3):
    _, aspect_d, _, _ = calc_differences_properties(data)
    print('------------\nAspect\n------------')
    print_table(get_center(aspect_d, window_size_list[-1] + 2))

    for window_size in [3, 5, 7]:

        _, aspect_2, _, _ = calc_second_order_properties(data, window_size, data_size)
        _, aspect_4, _, _ = calc_fourth_order_properties(data, window_size, data_size)

        print(f'\nWindow Size {window_size}\n')
        print('Differences')
        print_table(get_center(aspect_d, show_window))
        print('Second Order')
        print_table(get_center(aspect_2, show_window))
        print('Fourth Order')
        print_table(get_center(aspect_4, show_window))

def show_profilec(data, window_size_list, show_window=3):
    _, _, profilec_d, _ = calc_differences_properties(data)
    print('------------\nProfilkrümmung\n------------')
    print_table(get_center(profilec_d, window_size_list[-1] + 2))

    for window_size in window_size_list:

        _, _, profilec_2, _ = calc_second_order_properties(data, window_size, data_size)
        _, _, profilec_4, _ = calc_fourth_order_properties(data, window_size, data_size)

        print(f'\nWindow Size {window_size}\n')
        print('Differences')
        print_table(get_center(profilec_d, show_window))
        print('Second Order')
        print_table(get_center(profilec_2, show_window))
        print('Fourth Order')
        print_table(get_center(profilec_4, show_window))

def show_planc(data, window_size_list, show_window=3):
    _, _, _, planc_d = calc_differences_properties(data)
    print('------------\nPlankrümmung\n------------')
    print_table(get_center(planc_d, window_size_list[-1] + 2))

    for window_size in window_size_list:

        _, _, _, planc_2 = calc_second_order_properties(data, window_size, data_size)
        _, _, _, planc_4 = calc_fourth_order_properties(data, window_size, data_size)

        print(f'\nWindow Size {window_size}\n')
        print('Differences')
        print_table(get_center(planc_d, show_window))
        print('Second Order')
        print_table(get_center(planc_2, show_window))
        print('Fourth Order')
        print_table(get_center(planc_4, show_window))

############################################################################

data_size = 23
data = get_real_data(data_size,data_size).astype(np.float32)
window_size_list = [3, 21]
window_size_list_c = [3, 21]

show_slope(data, window_size_list)
show_aspect(data, window_size_list)
show_profilec(data, window_size_list_c)
show_planc(data, window_size_list_c)