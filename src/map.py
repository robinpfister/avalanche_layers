import glob
import os
import json
import requests
from osgeo import gdal, ogr, osr
import numpy as np
import time
from matplotlib import pyplot
import matplotlib
from matplotlib.colors import Normalize
import gitlab
from enum import Enum
import xml.etree.ElementTree as ET
import shutil
from datetime import date

from layer import Layer

matplotlib.use('tkagg')

class Region(Enum):
    BAVARIA = 'bavaria'
    SWITZERLAND = 'switzerland'
    TYROL = 'tyrol'

class Daytime(Enum):
    EARLY = 'preprocess_layer/danger_early.tif'
    LATE = 'preprocess_layer/danger_late.tif'

class Map():
    
    def __init__(self, layerCalculatorFactory):
        gdal.UseExceptions()

        self.layer_data_directory = 'layer_data'
        self.input_data_directory = 'input_data'
        
        self.working_daytime = None
        self.working_region = None
        self.working_region_directory = None

        self.height_layer_path = None
        self.avalanche_report_region_layer_path = None
        self.microregions_path = None
        self.standardized_avalanche_report_path = None
        self.raw_avalanche_report_path = None

        self.working_x_length = None
        self.working_y_length = None
        self.working_projection = None
        self.working_geotransform = None
        self.working_srs = None

        today = date.today()
        date_str = today.strftime("%Y-%m-%d")
        self.avalanche_report_url = {Region.BAVARIA : f'https://static.lawinen-warnung.eu/bulletins/{date_str}/{date_str}_DE-BY_de_CAAMLv6.xml',
                                     Region.SWITZERLAND : "https://aws.slf.ch/api/bulletin/caaml/v4/de/geojson?activeAt=2025-04-15T01:00:00%2B02:00",
                                     Region.TYROL : "https://static.avalanche.report/bulletins/latest/EUREGIO_de_CAAMLv6.xml"}

        self.avalanche_report_microregions = {Region.BAVARIA : ['DE-BY'],
                                              Region.SWITZERLAND : None,
                                              Region.TYROL : ['AT-07', 'IT-32-BZ', 'IT-32-TN']}

        self.layerCalculatorFactory = layerCalculatorFactory

##### BASE FUNKTIONS #####

    def register_new_data(self):
        self.delete_layer_data()
        for region in Region:
            self.set_working_region(region)
            if self.input_data_available():
                self.create_directory(f'{self.layer_data_directory}/{self.working_region_directory}')
                self.create_directory(f'{self.layer_data_directory}/{self.working_region_directory}/report_data')
                self.create_directory(f'{self.layer_data_directory}/{self.working_region_directory}/preprocess_layer/coefficients')
                self.create_directory(f'{self.layer_data_directory}/{self.working_region_directory}/preprocess_layer/rsa')
                self.preprocess_dgm()
                self.set_tif_properties()
                self.preprocess_avalanche_report()

    def set_working_region(self, region):
        self.working_region = region
        self.working_region_directory = region.value
        
        self.height_layer_path = f'{self.layer_data_directory}/{self.working_region_directory}/{Layer.HEIGHT.value}'
        self.avalanche_report_region_layer_path = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/avalanche_report_region.tif'
        
        self.microregions_path = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/microregions.geojson'
        
        self.standardized_avalanche_report_path = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/standardized_avalanche_report.json'
        self.raw_avalanche_report_path = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/raw_avalanche_report'
        if region == Region.SWITZERLAND:
            self.raw_avalanche_report_path = f'{self.raw_avalanche_report_path}.json'
        elif region == Region.BAVARIA or region == Region.TYROL:
            self.raw_avalanche_report_path = f'{self.raw_avalanche_report_path}.xml'

        if os.path.exists(self.height_layer_path):
            self.set_tif_properties()

    def set_working_daytime(self, daytime):
        self.working_daytime = daytime

    def set_tif_properties(self):
        array, geotransform, projection, srs = self.get_tif_data(self.height_layer_path)
        self.working_x_length = array.shape[1]
        self.working_y_length = array.shape[0]
        self.working_geotransform = geotransform
        self.working_projection = projection
        self.working_srs = srs

    def delete_layer(self, layertype):
        if type(layertype) == Layer:
            self.delete_file(f'{self.layer_data_directory}/{self.working_region_directory}/{layertype.value}')
        else:
            for inside_layertype in layertype:
                self.delete_file(f'{self.layer_data_directory}/{self.working_region_directory}/{inside_layertype.value}')

##### DIRECTORY MANAGEMENT #####

    def delete_layer_data(self):
        if os.path.exists(self.layer_data_directory) and os.path.isdir(self.layer_data_directory):
                shutil.rmtree(self.layer_data_directory)

    def delete_file(self, path):
        os.remove(path)

    def create_directory(self, path):
        os.makedirs(path)

    def input_data_available(self):
        input_data_path = f'{self.input_data_directory}/{self.working_region_directory}'
        return os.path.isdir(input_data_path) and len(os.listdir(input_data_path)) > 0

##### TIF/JSON HANDLING FUNKTIONS #####

    def get_tif_data(self, path):
        file = gdal.Open(path)
        array = file.ReadAsArray()
        geotransform = file.GetGeoTransform()
        projection = file.GetProjection()
        srs = file.GetSpatialRef()
        file = None
        return array, geotransform, projection, srs

    def create_tif(self, path, array, geotransform, projection):
        driver = gdal.GetDriverByName('GTiff')

        y_length, x_length = array.shape

        raster = driver.Create(path,
                                      x_length,
                                      y_length,
                                      1,
                                      eType = gdal.GDT_Float32)

        raster.SetProjection(projection)
        raster.SetGeoTransform(geotransform)

        band = raster.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.WriteArray(array)          
        band.FlushCache()
        band.ComputeStatistics(False)

    def get_geojson_data(self, path):
        file = gdal.OpenEx(path, 0)
        layer = file.GetLayer()

        srs = layer.GetSpatialRef()
        features = []
        for feature in layer:
            features.append(feature)

        file = None

        return features, srs

    def create_geojson(self, path, name, srs, feature_list):
        driver = ogr.GetDriverByName('GeoJSON')
        file = driver.CreateDataSource(path)
        layer = file.CreateLayer(name, srs=srs)
        for feature in feature_list:
            layer.CreateFeature(feature)
        file = None

##### DGM PREPROCESS STUFF #####

    def preprocess_dgm(self):
        self.create_height_layer()
        self.standardize_height_layer()
    
    def create_height_layer(self):
        tif_files = glob.glob(f'{self.input_data_directory}/{self.working_region_directory}/*.tif') # get list of all hight_layers in input_data/region folder
        gdal.Warp(self.height_layer_path, tif_files, format="GTiff", srcNodata=-9999, dstNodata=-9999) # join all hight_layers to one big height_layer

    def standardize_height_layer(self):
        if self.working_region == Region.TYROL:
            self.cut_edge_layer(self.height_layer_path) # cut edge row/column
            self.downscale_layer(self.height_layer_path, 2) # downscale factor 2 (0.5m -> 1m)
        elif self.working_region == Region.SWITZERLAND:
            self.downscale_layer(self.height_layer_path, 2) # downscale factor 2 (0.5m -> 1m)

    def cut_edge_layer(self, path):
        layer_array, geotransform, projection, _ = self.get_tif_data(path)
        layer_array = layer_array[0:-1, 0:-1]
        self.delete_file(path)
        self.create_tif(path, layer_array, geotransform, projection)

    def downscale_layer(self, path, scale_factor):
        layer_array, geotransform, projection, _ = self.get_tif_data(path)
        y_length, x_length = layer_array.shape
        layer_array = layer_array.reshape(y_length//2, 2, x_length//2, 2).mean(axis=(1, 3)) # combine 4 pixels to one
        geotransform = (geotransform[0],
                        geotransform[1]*scale_factor,
                        geotransform[2],
                        geotransform[3],
                        geotransform[4],
                        geotransform[5]*scale_factor) # scale pixel width
        self.delete_file(path)
        self.create_tif(path, layer_array, geotransform, projection)

##### AVALANCHE REPORT PREPROCESS STUFF #####

    def preprocess_avalanche_report(self):
        if self.working_region in [Region.BAVARIA, Region.TYROL]:
            self.create_microregion_definition()
        self.pull_avalanche_report()
        self.standardize_avalanche_report()
        self.burn_geometries_in_raster()
        self.create_danger_layer()

    def create_microregion_definition(self):
            self.pull_microregions()
            self.convert_geojson_srs(self.microregions_path)

    def pull_avalanche_report(self):
        response = requests.get(self.avalanche_report_url[self.working_region])
        if response.status_code == 200:
            data = response.content
            with open(f'{self.raw_avalanche_report_path}', 'wb') as file:
                file.write(data)
        else:
            print(response.content)
            print('Download of avalanche report doesn\'t work')
        
    def standardize_avalanche_report(self):
        if self.working_region == Region.BAVARIA or self.working_region == Region.TYROL:
            self.create_standardized_avalanche_report_CAAMLV6()
        elif self.working_region == Region.SWITZERLAND:
            self.create_standardized_avalanche_report_switzerland()
            self.convert_geojson_srs(self.standardized_avalanche_report_path)
            
    def pull_microregions(self):
        gl = gitlab.Gitlab("https://gitlab.com")
        project = gl.projects.get(25330421)

        for region in self.avalanche_report_microregions[self.working_region]:
            microregion_file = project.files.get(f'public/micro-regions/{region}_micro-regions.geojson.json', 'master')
            file_data = microregion_file.decode()

            if os.path.exists(self.microregions_path): # check if microregion file already exist and add new microregion to the file
                new_data = json.loads(file_data)
                new_features = new_data['features']
                with open(self.microregions_path, "r") as file:
                    all_data = json.load(file)
                    all_features = all_data['features']
                    for new_feature in new_features:
                        all_features.append(new_feature)
                    all_data['features'] = all_features
                with open(self.microregions_path, "w") as file:
                    json.dump(all_data, file, ensure_ascii=False, indent=4)
            else: # just save data - iteration 0
                with open(self.microregions_path, "wb") as file:
                    file.write(file_data)

    def convert_geojson_srs(self, path):
        src_features, src_srs = self.get_geojson_data(path)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        dst_srs = self.working_srs # srs of tif
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        coordinates_transformation = osr.CoordinateTransformation(src_srs, dst_srs)
        
        dst_features = []
        for src_feature in src_features:
            src_feature.GetGeometryRef().Transform(coordinates_transformation) # transform geometry to dst srs

            dst_features.append(src_feature)
        
        self.delete_file(path)
        self.create_geojson(path, 'test', dst_srs, dst_features)

    def burn_geometries_in_raster(self):

        features, srs = self.get_geojson_data(self.standardized_avalanche_report_path)

        driver = gdal.GetDriverByName('GTiff')
        mask_raster = driver.Create(self.avalanche_report_region_layer_path,
                                        self.working_x_length,
                                        self.working_y_length,
                                        1,
                                        eType = gdal.GDT_Float32)
        mask_raster.SetProjection(self.working_projection)
        mask_raster.SetGeoTransform(self.working_geotransform)

        driver = gdal.GetDriverByName("Memory")
        for feature in features:
            temp_file = driver.CreateDataSource('')
            temp_layer = temp_file.CreateLayer('temp_layer', srs=srs, geom_type=ogr.wkbPolygon)
            temp_layer.CreateFeature(feature)

            gdal.RasterizeLayer(mask_raster, [1], temp_layer, burn_values=[feature.GetFID()])

        mask_raster.FlushCache()
        mask_raster = None

    def create_standardized_avalanche_report_CAAMLV6(self): 
        tag_prefix = '{http://caaml.org/Schemas/V6.0/Profiles/BulletinEAWS}'
        tree = ET.parse(self.raw_avalanche_report_path)
        root = tree.getroot()

        features = []
        for i, bulletin in enumerate(root):
            # Get danger properties
            dangers = []
            for danger_rating in bulletin.findall(f'{tag_prefix}dangerRating'):
                main_value = danger_rating.find(f'{tag_prefix}mainValue')
                main_value = main_value.text
                elevation = danger_rating.find(f'{tag_prefix}elevation')
                if elevation != None:
                    lower_bound = elevation.find(f'{tag_prefix}lowerBound')
                    upper_bound = elevation.find(f'{tag_prefix}upperBound')
                    if lower_bound != None:
                        lower_bound = lower_bound.text
                    else:
                        lower_bound = None
                    if upper_bound != None:
                        upper_bound = upper_bound.text
                    else:
                        upper_bound = None
                else:
                    lower_bound = None
                    upper_bound = None
                time_slot = danger_rating.find(f'{tag_prefix}validTimePeriod')
                time_slot = time_slot.text
                danger = {'main_value' : main_value,
                          'sub_value' : None,
                          'lower_bound' : lower_bound,
                          'upper_bound' : upper_bound,
                          'time_slot' : time_slot}
                dangers.append(danger)

            # Get avalanche problem properties
            problems = []
            for avalanche_problem in bulletin.findall(f'{tag_prefix}avalancheProblem'):
                problem_type = avalanche_problem.find(f'{tag_prefix}problemType')
                problem_type = problem_type.text
                elevation = avalanche_problem.find(f'{tag_prefix}elevation')
                lower_bound = elevation.find(f'{tag_prefix}lowerBound')
                upper_bound = elevation.find(f'{tag_prefix}upperBound')
                if lower_bound != None:
                    lower_bound = lower_bound.text
                else:
                    lower_bound = None
                if upper_bound != None:
                    upper_bound = upper_bound.text
                else:
                    upper_bound = None
                aspects = []
                for aspect in avalanche_problem.findall(f'{tag_prefix}aspect'):
                    aspects.append(aspect.text)
                time_slot = avalanche_problem.find(f'{tag_prefix}validTimePeriod')
                time_slot = time_slot.text
                problem = {'type' : problem_type,
                           'lower_bound' : lower_bound,
                           'upper_bound' : upper_bound,
                           'aspects' : aspects,
                           'time_slot' : time_slot}
                problems.append(problem)
            
            # Get region properties
            regions = []
            for region in bulletin.findall(f'{tag_prefix}region'):
                region_id = region.get('regionID')
                regions.append(region_id)
            
            # Convert region_ids to polygones
            microregion_data = None
            file = open(self.microregions_path, "r", encoding="utf-8")
            microregion_data = json.load(file)
            file.close()

            polygones = []
            mr = microregion_data['features']
            for micro_region in mr:
                if micro_region['properties']['id'] in regions:
                    if 'end_date' in micro_region['properties'] and micro_region['properties']['end_date'] != None:
                        continue
                    for polygon in micro_region['geometry']['coordinates']:
                        polygones.append(polygon)

            # Create feature
            feature = {'type' : 'Feature',
                       'properties' :   {'id' : i,
                                         'Danger' : dangers,
                                         'Problems' : problems,
                                         'Patterns' : None},
                        'geometry' :    {'type' : 'MultiPolygon',
                                         'coordinates' : polygones}}
            features.append(feature)
        
        file = open(self.microregions_path, "r", encoding="utf-8")
        microregion_data = json.load(file)
        file.close()
        crs = microregion_data['crs']['properties']['name']
        # Create GeoJSON structure
        standardized_avalanche_report_dict = {"type": "FeatureCollection",
                                              "name": "standardized_avalanche_report",
                                              "crs": { "type": "name", "properties": { "name": crs } },
                                              "features": features}
    
        with open(self.standardized_avalanche_report_path, 'w') as f:
            json.dump(standardized_avalanche_report_dict, f, indent= 4)

    def create_standardized_avalanche_report_switzerland(self): 
        
        driver = ogr.GetDriverByName('GeoJSON')
        src_file = driver.Open(self.raw_avalanche_report_path, 0)  # 0 bedeutet nur Lesen
        src_layer = src_file.GetLayer()

        dst_file = driver.CreateDataSource(self.standardized_avalanche_report_path)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)
        dst_layer = dst_file.CreateLayer('standardized_avalanche_report', srs=dst_srs)
        dst_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
        dst_layer.CreateField(ogr.FieldDefn("Danger", ogr.OFTString))
        dst_layer.CreateField(ogr.FieldDefn("Problems", ogr.OFTString))
        dst_layer.CreateField(ogr.FieldDefn("Patterns", ogr.OFTString))


        for i, bulletin in enumerate(src_layer):
            dangers = []
            danger_ratings = bulletin.GetField("dangerRatings")
            danger_ratings = json.loads(danger_ratings)
            for danger_rating in danger_ratings: # elevation?
                main_value = danger_rating['mainValue']
                if 'customData' in danger_rating and 'CH' in danger_rating['customData'] and 'subdivision' in danger_rating['customData']['CH']:
                    sub_value = danger_rating['customData']['CH']['subdivision']
                else:
                    sub_value = None
                time_slot = danger_rating['validTimePeriod']
                danger = {'main_value' : main_value,
                          'sub_value' : sub_value,
                          'lower_bound' : None,
                          'upper_bound' : None,
                          'time_slot' : time_slot}
                dangers.append(danger)
            
            problems = []
            avalanche_problems = bulletin.GetField("avalancheProblems")
            avalanche_problems = json.loads(avalanche_problems)
            for avalanche_problem in avalanche_problems:
                problem_type = avalanche_problem['problemType']
                if 'elevation' in avalanche_problem:
                    if 'lower_bound' in avalanche_problem['elevation']:
                        lower_bound = avalanche_problem['elevation']['lower_bound']
                    else:
                        lower_bound = None
                    if 'upper_bound' in avalanche_problem['elevation']:
                        upper_bound = avalanche_problem['elevation']['lower_bound']
                    else:
                        upper_bound = None
                else:
                    lower_bound = None
                    upper_bound = None

                if 'aspects' in avalanche_problem:
                    aspects = avalanche_problem['aspects']
                else:
                    aspects = None
                
                time_slot = avalanche_problem['validTimePeriod']

                problem = {'type' : problem_type,
                           'lower_bound' : lower_bound,
                           'upper_bound' : upper_bound,
                           'aspects' : aspects,
                           'time_slot' : time_slot}
                problems.append(problem)
            
            geom = bulletin.GetGeometryRef().Clone()

            out_feature = ogr.Feature(dst_layer.GetLayerDefn())
            out_feature.SetGeometry(geom)
            out_feature.SetField("id", i)
            out_feature.SetField("Danger", json.dumps(dangers))
            out_feature.SetField("Problems", json.dumps(problems))
            out_feature.SetField("Patterns", json.dumps(None))

            dst_layer.CreateFeature(out_feature)
            
        src_file = None
        dst_file = None

    def create_danger_layer(self):
        features, _ = self.get_geojson_data(self.standardized_avalanche_report_path)

        earlier_list = []
        later_list = []
        for feature in features:
            danger_ratings = feature.GetField("Danger")
            danger_ratings = json.loads(danger_ratings)
            max_all_day = 0
            max_earlier = 0
            max_later = 0
            for danger_rating in danger_ratings:
                if danger_rating['time_slot'] == 'all_day':
                    all_day = self.convert_mainValue(danger_rating['main_value'])
                    all_day = all_day + self.convert_subValue(danger_rating['sub_value'])
                    if all_day > max_all_day:
                        max_all_day = all_day
                if danger_rating['time_slot'] == 'earlier':
                    earlier = self.convert_mainValue(danger_rating['main_value'])
                    earlier = earlier + self.convert_subValue(danger_rating['sub_value'])
                    if earlier > max_earlier:
                        max_earlier = earlier
                elif danger_rating['time_slot'] == 'later':
                    later = self.convert_mainValue(danger_rating['main_value'])
                    later = later + self.convert_subValue(danger_rating['sub_value'])
                    if later > max_later:
                        max_later = later
            if max_all_day != 0:
                earlier_list.append(max_all_day)
                later_list.append(max_all_day)
            else:
                earlier_list.append(max_earlier)
                later_list.append(max_later)
        
        region_layer, _, _, _ = self.get_tif_data(self.avalanche_report_region_layer_path)
        danger_layer_early = np.full(region_layer.shape, np.nan)
        danger_layer_late = np.full(region_layer.shape, np.nan)

        for i in range(len(features)):
            danger_layer_early = np.where(region_layer == np.float32(i), earlier_list[i], danger_layer_early)
        for i in range(len(features)):
            danger_layer_late = np.where(region_layer == np.float32(i), later_list[i], danger_layer_late)
        
        self.save_layer([Layer.DANGER_EARLY, Layer.DANGER_LATE], [danger_layer_early, danger_layer_late])

        if self.working_daytime != None:
            danger_array, _, _, _ = self.get_tif_data(f'{self.layer_data_directory}/{self.working_region_directory}/{self.working_daytime.value}')
            self.save_layer([Layer.SITUATION_RISK], [danger_array])

    def convert_mainValue(self, mainValue):
        mainValueInt = 0
        if mainValue == 'low':
            mainValueInt = 1
        elif mainValue == 'moderate':
            mainValueInt = 2
        elif mainValue == 'considerable':
            mainValueInt = 3
        elif mainValue == 'high':
            mainValueInt = 4
        return mainValueInt
    
    def convert_subValue(self, subValue):
        mainValueInt = 0
        if subValue == 'minus':
            mainValueInt = -0.33
        elif subValue == 'neutral':
            mainValueInt = 0
        elif subValue == 'plus':
            mainValueInt = 0.33
        return mainValueInt

##### CALCULATE #####

    def calculateLayer(self, layertype):
        layerCalculator = self.layerCalculatorFactory.get_calculator(layertype)
        layers = self.get_required_layers_dict(layerCalculator.required_layers)
        
        start = time.time()
        layer_arrays = layerCalculator.calculate(layers)
        ende = time.time()
        print(f'[LOG] Berechnungzeit {layertype}: {round(ende-start, 2)}s')

        self.save_layer(layertype, layer_arrays)

    def get_required_layers_dict(self, required_layertypes):
        layers_dict = {}
        for required_layertype in required_layertypes:
            required_layer_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/{required_layertype.value}'

            if os.path.isfile(required_layer_filepath): # layer is already available -> take the array data and copy to layers dict
                file = gdal.Open(required_layer_filepath)
                array = np.array(file.ReadAsArray(), dtype=np.float32)
                layers_dict[required_layertype] = array
                file = None
            else: # layer is not available -> recurive execution of calculateLayer() for the neccessary layer
                if type(required_layertype) == Layer: # layertype != preprocessed data
                    self.calculateLayer(required_layertype)
                else: # layertype == preprocessed data
                    self.calculateLayer(type(required_layertype))
                file = gdal.Open(required_layer_filepath)
                array = np.array(file.ReadAsArray(), dtype=np.float32)
                layers_dict[required_layertype] = array
                file = None
        return layers_dict

    def save_layer(self, layertypes, layer_arrays):
        if type(layertypes) == Layer: # layertype != preprocessed data
            layertypes = [layertypes]
            layer_arrays = [layer_arrays]

        for i, layertype in enumerate(layertypes):
            layer_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/{layertype.value}'
            self.create_tif(layer_filepath, layer_arrays[i], self.working_geotransform, self.working_projection)

##### SHOW #####

    def showLayer(self, layertypes):

        min_max_mapping = { Layer.PROFILE_CURVATURE :   {'min' : -0.3,
                                                         'max' : 0.3},
                            Layer.PLAN_CURVATURE :      {'min' : -0.3,
                                                         'max' : 0.3}}
        fig = pyplot.figure(figsize=(10, 8))

        for i, layertype in enumerate(layertypes):
            layer_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/{layertype.value}'

            ax = fig.add_subplot(1, len(layertypes), i+1)
            file = gdal.Open(layer_filepath)
            layer_array = file.ReadAsArray()
            if layertype in min_max_mapping:
                min  = min_max_mapping[layertype]['min']
                max  = min_max_mapping[layertype]['max']
            else:
                manipulated_array = np.where(np.isnan(layer_array), 0, layer_array)
                min = manipulated_array.min()
                max = manipulated_array.max()
            
            ax.imshow(layer_array, cmap = 'Dark2', vmin = min, vmax = max) # extent=[660000, 670000, 5270000, 5280000]
            norm = Normalize(min, max)
            m = pyplot.cm.ScalarMappable(norm=norm, cmap= 'viridis')
            m.set_array([])
            pyplot.colorbar(m, ax=ax, orientation="horizontal")
        pyplot.show()

    def show3D(self, layertypes, cmaps, values, area):
        fig = pyplot.figure(figsize=(10, 8))

        x = np.linspace(0, area[3]-1, area[3])
        y = np.linspace(0, area[2]-1, area[2])
        X,Y = np.meshgrid(x, y)

        for i, layertype in enumerate(layertypes):
            layer_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/{layertype.value}'
            file = gdal.Open(layer_filepath)
            layer_array = file.ReadAsArray()
            color_dimension = np.copy(layer_array[area[0]:area[0]+area[2], area[1]:area[1]+area[3]])
            minn = values[i][0]
            maxx = values[i][1]
            if minn == None:
                minn = color_dimension.min()
            if maxx == None:
                maxx = color_dimension.max()
            norm = Normalize(minn, maxx)
            m = pyplot.cm.ScalarMappable(norm=norm, cmap=cmaps[i])
            m.set_array([])
            fcolor = m.to_rgba(color_dimension)
                    
            ax = fig.add_subplot(1, len(layertypes), i+1, projection='3d')
            file = gdal.Open(self.height_layer_path)
            height_array = file.ReadAsArray()
            Z = height_array[area[0]:area[0]+area[2], area[1]:area[1]+area[3]]
            ax.plot_surface(X, Y, Z, facecolors=fcolor)
            pyplot.colorbar(m, ax=ax, orientation="horizontal")
        pyplot.show()
