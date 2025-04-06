from osgeo import gdal
from matplotlib import pyplot
import json

file = gdal.Open(f'layer_data/bavaria/report_data/avalanche_report_region.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()

file = gdal.Open(f'layer_data/bavaria/danger_early.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()

file = gdal.Open(f'layer_data/bavaria/danger_late.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()