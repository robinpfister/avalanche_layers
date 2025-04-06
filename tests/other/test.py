from osgeo import gdal
from matplotlib import pyplot
import json

file = gdal.Open(f'layer_data/bavaria/report_data/avalanche_report_region_layer.tif')
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

file = gdal.Open(f'layer_data/tyrol/report_data/avalanche_report_region_layer.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()

file = gdal.Open(f'layer_data/tyrol/danger_early.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()

file = gdal.Open(f'layer_data/tyrol/danger_late.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()

file = gdal.Open(f'layer_data/switzerland/report_data/avalanche_report_region_layer.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()

file = gdal.Open(f'layer_data/switzerland/danger_early.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()

file = gdal.Open(f'layer_data/switzerland/danger_late.tif')
layer_array = file.ReadAsArray()
pyplot.imshow(layer_array)
pyplot.show()