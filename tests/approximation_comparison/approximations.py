import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from osgeo import gdal
import time

print(np.linspace(-2, 2, 5))

def calc_second_order(heights, window_size, grid_size):
    F = np.zeros((window_size * window_size, 6))
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

    coeffs = np.zeros((grid_size, grid_size, 6))
    
    for y in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
        for x in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
            Z = heights[y - int((window_size-1)/2) : y + int((window_size-1)/2) + 1, 
                        x - int((window_size-1)/2) : x + int((window_size-1)/2) + 1].ravel().astype(np.float32)
            coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  
    
    mid = grid_size // 2
    A, B, C, D, E, F = coeffs[mid, mid, 0], coeffs[mid, mid, 1], coeffs[mid, mid, 2], coeffs[mid, mid, 3], coeffs[mid, mid, 4], coeffs[mid, mid, 5]
    
    slope = ((D ** 2) + (E ** 2)) ** (1/2)
    slope = np.arctan((slope)) * (180 / np.pi)

    aspect = np.arctan2(-E, -D)
    aspect = (aspect * (180 / np.pi)) + 180

    profilec = ((-2 * ((A * (D ** 2)) + (B * (E ** 2)) + (C * D * E))) / (((D ** 2) + (E ** 2)) * ((1 + (D ** 2) + (E ** 2)) ** (3/2))))
    planc = ((-2 * ((B * (D ** 2)) + (A * (E ** 2)) - (C * D * E))) / (((D ** 2) + (E ** 2)) ** (3/2)))

    return (lambda x, y: (A*(x**2)) + (B*(y**2)) + (C*(x*y)) + (D*x) + (E*y) + F), round(slope, 2), round(aspect, 2), round(profilec, 2), round(planc, 2)

def calc_fourth_order(heights, window_size, grid_size):
    F = np.zeros((window_size * window_size, 9))
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

    coeffs = np.zeros((grid_size, grid_size, 9))
    
    for y in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
        for x in range(int((window_size-1)/2), grid_size - int((window_size-1)/2)):
            Z = heights[y - int((window_size-1)/2) : y + int((window_size-1)/2) + 1, 
                        x - int((window_size-1)/2) : x + int((window_size-1)/2) + 1].ravel().astype(np.float32)
            coeffs[y, x] = np.dot(XtX_inv_Xt, Z)  
    
    mid = grid_size // 2
    A, B, C, D, E, F, G, H, I = coeffs[mid, mid, 0], coeffs[mid, mid, 1], coeffs[mid, mid, 2], coeffs[mid, mid, 3], coeffs[mid, mid, 4], coeffs[mid, mid, 5], coeffs[mid, mid, 6], coeffs[mid, mid, 7], coeffs[mid, mid, 8] 
    
    slope = ((G ** 2) + (H ** 2)) ** (1/2)
    slope = np.arctan((slope)) * (180 / np.pi)

    aspect = np.arctan2(-H, -G)
    aspect = (aspect * (180 / np.pi)) + 180

    profilec = ((-2 * ((D * (G ** 2)) + (E * (H ** 2)) + (F * G * H))) / (((G ** 2) + (H ** 2)) * ((1 + (G ** 2) + (H ** 2)) ** (3/2))))
    planc = ((-2 * ((D * (H ** 2)) + (E * (G ** 2)) - (F * G * H))) / (((G ** 2) + (H ** 2)) ** (3/2)))
    
    return (lambda x, y: (A*(x**2*y**2)) + (B*(y*x**2)) + (C*(x*y**2)) + (D*(x**2)) + (E*(y**2)) + (F*(x*y)) + (G*x) + (H*y) + I), round(slope, 2), round(aspect, 2), round(profilec, 2), round(planc, 2)

def generate_data(rows, cols):
    """Erstellt ein 2D-Array basierend auf der Funktion f(x, y) = x^2 * y^2."""
    x = np.arange(cols) - cols // 2
    y = np.arange(rows) - rows // 2
    X, Y = np.meshgrid(x, y)
    return 10-(X**2 +Y**2)

def get_real_data(rows, cols):
    file = gdal.Open(f'data_layer/data_deutschland/height.tif')
    data = file.ReadAsArray()[5057-int((rows-1)/2):5057+int((rows-1)/2) +1, 3027-int((cols-1)/2):3027+int((cols-1)/2)+ 1]
    #[2057-int((rows-1)/2):2057+int((rows-1)/2) +1, 8027-int((cols-1)/2):8027+int((cols-1)/2)+ 1]
    #[4057-int((rows-1)/2):4057+int((rows-1)/2) +1, 3027-int((cols-1)/2):3027+int((cols-1)/2)+ 1]
    return data -1482

def plot_3d_bar_chart_with_surface(data, func1, func2, xlim, ylim, bar_area):
    resolution = int(20 *xlim[1])

    fig = plt.figure(figsize=(14, 7))

    # Plot für die erste Funktion
    ax1 = fig.add_subplot(121, projection='3d')
    X = np.linspace(xlim[0], xlim[1], resolution)
    Y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(X, Y)
    Z1 = func1[0](X, Y)

    # Dimensionen des Arrays für Balken
    ny, nx = data.shape
    x, y = np.meshgrid(np.arange(bar_area) - bar_area/2, np.arange(bar_area) - bar_area/2)
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)

    # Werte als Höhen der Balken
    dx = dy = 1  # Balkenbreite
    dz = data[int(((np.shape(data)[0]) - bar_area) / 2) : bar_area + int(((np.shape(data)[0]) - bar_area) / 2), int(((np.shape(data)[0]) - bar_area) / 2) : bar_area + int(((np.shape(data)[0]) - bar_area) / 2)].flatten()
    minimum = np.min(dz)
    if minimum < 0:
        dz = dz - minimum
    
    wndw_size = int(-xlim[0] + xlim[1])

    v = np.linspace(-(wndw_size-1)/2, (wndw_size-1)/2, wndw_size)
    b = np.linspace(-(wndw_size-1)/2, (wndw_size-1)/2, wndw_size)
    v, b = np.meshgrid(v, b)
    O = func1[0](v, b)
    d = O - data[int(((np.shape(data)[0]) - wndw_size) / 2) : wndw_size + int(((np.shape(data)[0]) - wndw_size) / 2), int(((np.shape(data)[0]) - wndw_size) / 2) : wndw_size + int(((np.shape(data)[0]) - wndw_size) / 2)]
    d = d ** 2
    d = np.average(d) 

    # Oberfläche für die erste Funktion
    ax1.plot_wireframe(X, Y, Z1, alpha=1, color='red', rstride=int(resolution / (ylim[1] - ylim[0])), cstride=int(resolution / (xlim[1] - xlim[0])), edgecolors='b', linewidth=3)
    ax1.bar3d(x, y, z, dx, dy, dz, alpha=0.15, shade=True, color='tan', edgecolor='black')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Polynom 2. Ordnung')
    ax1.text2D(0.05, 0.9, f'Differenz: {d}\n\nHangneigung: {func1[1]}°\n\nExposition: {func1[2]}°\n\nProfilkrümmung: {func1[3]}\n\nPlankrümmung: {func1[4]}', horizontalalignment='left',
     verticalalignment='center', transform=ax1.transAxes)
    ax1.view_init(elev=20, azim=75)

    # Plot für die zweite Funktion
    ax2 = fig.add_subplot(122, projection='3d')
    Z2 = func2[0](X, Y)

    # # Dimensionen des Arrays für Balken
    # ny, nx = data.shape
    # x, y = np.meshgrid(np.arange(nx) - nx / 2, np.arange(ny) - ny / 2)
    # x = x.flatten()
    # y = y.flatten()
    # z = np.zeros_like(x)

    # # Werte als Höhen der Balken
    # dz = data.flatten()
   

    v = np.linspace(-(wndw_size-1)/2, (wndw_size-1)/2, wndw_size)
    b = np.linspace(-(wndw_size-1)/2, (wndw_size-1)/2, wndw_size)
    v, b = np.meshgrid(v, b)
    O = func2[0](v, b)
    d = O - data[int(((np.shape(data)[0]) - wndw_size) / 2) : wndw_size + int(((np.shape(data)[0]) - wndw_size) / 2), int(((np.shape(data)[0]) - wndw_size) / 2) : wndw_size + int(((np.shape(data)[0]) - wndw_size) / 2)]
    d = d ** 2
    d = np.average(d)

    # Oberfläche für die zweite Funktion
    ax2.plot_wireframe(X, Y, Z2, alpha=1, color='red', rstride=int(resolution / (ylim[1] - ylim[0])), cstride=int(resolution / (xlim[1] - xlim[0])), edgecolors='b', linewidth=3)
    ax2.bar3d(x, y, z, dx, dy, dz, alpha=0.15, shade=True, color='tan', edgecolor='black')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Polynom 4. Ordnung')
    ax2.text2D(0.05, 0.9, f'Differenz: {d}\n\nHangneigung: {func2[1]}°\n\nExposition: {func2[2]}°\n\nProfilkrümmung: {func2[3]}\n\nPlankrümmung: {func2[4]}', horizontalalignment='left',
     verticalalignment='center', transform=ax2.transAxes)
    ax2.view_init(elev=20, azim=75)

    plt.tight_layout()
    plt.show()

# Beispiel: 10x10 Array mit der Funktion x^2 * y^2 und einer Sinus-Oberfläche
# data = generate_data(5, 5)
data_size = 33
data = get_real_data(data_size, data_size)

plot_3d_bar_chart_with_surface(data, calc_second_order(data,3, data_size), calc_fourth_order(data,3, data_size),(-1.5, 1.5), (-1.5, 1.5), 5)

plot_3d_bar_chart_with_surface(data, calc_second_order(data,5, data_size), calc_fourth_order(data,5, data_size),(-2.5, 2.5), (-2.5, 2.5), 5)

plot_3d_bar_chart_with_surface(data, calc_second_order(data,7, data_size), calc_fourth_order(data,7, data_size),(-3.5, 3.5), (-3.5, 3.5), 7)

plot_3d_bar_chart_with_surface(data, calc_second_order(data,9, data_size), calc_fourth_order(data,9, data_size),(-4.5, 4.5), (-4.5, 4.5), 9)

plot_3d_bar_chart_with_surface(data, calc_second_order(data,11, data_size), calc_fourth_order(data,11, data_size),(-5.5, 5.5), (-5.5, 5.5), 11)