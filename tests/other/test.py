from osgeo import gdal
from matplotlib import pyplot
import json
import numpy as np
import matplotlib

matplotlib.use('TKAgg')

t = np.array([np.nan, 10, 9, 8, 11, np.nan, np.nan, np.nan])
print(np.nanmean(t))
t = np.array([0, 10, 9, 8, 11, 0, 0, 0])
print(np.nanmean(t))

# import numpy as np
# import matplotlib.pyplot as plt

# # Zeitachse
# t = np.linspace(0, 70, 500)

# # Parameter
# # K = 1     # Kapazitätsgrenze
# # r = 0.8     # Wachstumsrate
# # t0 = 34     # Wendepunkt

# K = 1     # Kapazitätsgrenze
# r = 0.8     # Wachstumsrate
# t0 = 34     # Wendepunkt

# # Standard logistische Funktion
# def logistic(t, K, r, t0):
#     return K / (1 + np.exp(-r * ((70 - t) - t0)))

# # Verallgemeinerte logistische Funktion (Richards-Kurve)
# def richards(t, K, r, t0, nu):
#     return K / (1 + np.exp(-r * (t - t0)))**(1/nu)

# # Gompertz-Funktion
# def gompertz(t, K, r, t0):
#     return K * np.exp(-np.exp(-r * (t - t0)))

# # Plot
# plt.figure(figsize=(10, 6))

# # Standard
# plt.plot(t, logistic(t, K, r, t0), label='Standard-Logistik', linestyle='--', color='black')

# # Richards mit verschiedenen nu
# nus = [1, 1.35, 2]
# colors = ['blue', 'green', 'orange']
# for nu, c in zip(nus, colors):
#     label = f'Richards (ν = {nu})'
#     plt.plot(t, richards(t, K, r, t0, nu), label=label, color=c)

# # Gompertz
# plt.plot(t, gompertz(t, K, r, t0), label='Gompertz', color='red')

# # Styling
# plt.title("Vergleich logistischer Wachstumsmodelle")
# plt.xlabel("Zeit t")
# plt.ylabel("f(t)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # file = gdal.Open(f'layer_data/bavaria/danger_late.tif')
# # layer_array = file.ReadAsArray()
# # pyplot.imshow(layer_array)
# # pyplot.show()