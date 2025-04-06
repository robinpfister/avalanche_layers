import glob
import os
import json
import requests
from osgeo import gdal, ogr, osr
import numpy as np
import time
from matplotlib import pyplot
from matplotlib.colors import Normalize
import gitlab
from enum import Enum
import xml.etree.ElementTree as ET
import shutil

from src.layers import Layers

class Region(Enum):
    BAVARIA = 'bavaria'
    SWITZERLAND = 'switzerland'
    TYROL = 'tyrol'

class Map():
    def __init__(self, layerCalculatorFactory):
        gdal.UseExceptions()

        self.layer_data_directory = 'layer_data'
        self.input_data_directory = 'input_data'
        self.working_region = None
        self.working_region_directory = None

        self.working_projection = None
        self.working_geotransform = None
        self.working_srs = None

        self.avalanche_report_url = {Region.BAVARIA : 'https://static.lawinen-warnung.eu/bulletins/latest/DE-BY_de_CAAMLv6.xml', #aktueller tag!!!!!!!!!!!!!
                                     Region.SWITZERLAND : "https://aws.slf.ch/api/bulletin/caaml/v4/de/geojson",
                                     Region.TYROL : "https://static.avalanche.report/bulletins/2025-04-04/2025-04-04_EUREGIO_de_CAAMLv6.xml"}#https://static.avalanche.report/bulletins/latest/EUREGIO_de_CAAMLv6.xml
        self.avalanche_report_microregions = {Region.BAVARIA : ['DE-BY'],
                             Region.SWITZERLAND : None,
                             Region.TYROL : ['AT-07', 'IT-32-BZ', 'IT-32-TN']}

        self.layerCalculatorFactory = layerCalculatorFactory

##### BASE FUNKTIONS #####

    def register_new_data(self):
        self.delete_layer_data()

        for region in Region: # loop through all supported regions
            self.set_working_region(region)
            self.create_directory(f'{self.layer_data_directory}/{self.working_region_directory}')
        
            self.preprocess_dgm()

            self.set_working_region(region)

            self.preprocess_avalanche_report()

    def set_working_region(self, region):
        if isinstance(region, Region):
            self.working_region = region
            self.working_region_directory = region.value

            if os.path.exists(f'{self.layer_data_directory}/{self.working_region.value}/height.tif'):
            # set working geotransform & projection
                file = gdal.Open(f'{self.layer_data_directory}/{self.working_region_directory}/height.tif')
                self.working_geotransform = tuple(file.GetGeoTransform()) #tuple(), str() nicht zwangsläufig nötig
                self.working_projection = str(file.GetProjection())
                self.working_srs = file.GetSpatialRef()

                file = None

##### DIRECTORY MANAGEMENT #####

    def delete_layer_data(self):
        if os.path.exists(self.layer_data_directory) and os.path.isdir(self.layer_data_directory):
                shutil.rmtree(self.layer_data_directory)

    def create_directory(self, path): # ?
        os.makedirs(path)

    def input_data_available(self):
        input_data_path = f'{self.input_data_directory}/{self.working_region_directory}'
        return os.path.isdir(input_data_path) and len(os.listdir(input_data_path)) > 0
        
##### DGM PREPROCESS STUFF #####

    def preprocess_dgm(self):
        if self.input_data_available():
            self.create_height_layer()
            self.standardize_height_layer()
    
    def create_height_layer(self):
        height_layer_path = f'{self.layer_data_directory}/{self.working_region_directory}/{Layers.HEIGHT.value}'

        tif_files = glob.glob(f'{self.input_data_directory}/{self.working_region_directory}/*.tif') # get list of all hight_layers in input_data/region folder
        gdal.Warp(height_layer_path, tif_files, format="GTiff", srcNodata=-9999, dstNodata=-9999) # join all hight_layers to one big height_layer

    def standardize_height_layer(self):
        height_layer_path = f'{self.layer_data_directory}/{self.working_region_directory}/{Layers.HEIGHT.value}'

        # extra editing for uniform height_layer
        if self.working_region == Region.TYROL:
            self.cut_edge_layer(height_layer_path) # cut edge row/column
            self.downscale_layer(height_layer_path, 2) # downscale factor 2 (0.5m -> 1m)
        elif self.working_region == Region.SWITZERLAND:
            self.downscale_layer(height_layer_path, 2) # downscale factor 2 (0.5m -> 1m)

    def cut_edge_layer(self, path): # CHECK
        
        file = gdal.Open(path)
        layer_array = file.ReadAsArray()
        geotransform = file.GetGeoTransform()
        projektion = file.GetProjection()
        file = None
        new_array = np.copy(layer_array[0:20000, 0:20000])
        rows, columns = np.shape(new_array)

        driver = gdal.GetDriverByName('GTiff')

        os.remove(path)

        output_raster = driver.Create(path,
                                            int(columns),
                                            int(rows),
                                            1,
                                            eType = gdal.GDT_Float32)

        output_raster.SetProjection(projektion)
        output_raster.SetGeoTransform(geotransform)

        band = output_raster.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.WriteArray(new_array)          
        band.FlushCache()
        band.ComputeStatistics(False)

    def downscale_layer(self, path, scale_factor): # CHECK
        file = gdal.Open(path)
        array = np.array(file.ReadAsArray(), dtype=np.float32)
        projektion = file.GetProjection()
        geotransform = file.GetGeoTransform()
        file = None

        def downscale_array(arr):
            h, w = arr.shape
            assert h % 2 == 0 and w % 2 == 0, "Die Array-Dimensionen müssen durch 2 teilbar sein!"
            downscaled = arr.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))
            return downscaled
        
        def scale_transform(transform, scale_factor):
            return (transform[0],
                    transform[1]*scale_factor,
                    transform[2],
                    transform[3],
                    transform[4],
                    transform[5]*scale_factor)

        geotransform = scale_transform(geotransform, scale_factor)
        array = downscale_array(array)
        
        driver = gdal.GetDriverByName('GTiff')
        rows, columns = np.shape(array)

        output_raster = driver.Create(path,
                                        int(columns),
                                        int(rows),
                                        1,
                                        eType = gdal.GDT_Float32)

        output_raster.SetProjection(projektion)
        output_raster.SetGeoTransform(geotransform)

        band = output_raster.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.WriteArray(array)          
        band.FlushCache()
        band.ComputeStatistics(False)

##### AVALANCHE REPORT PREPROCESS STUFF #####

    def preprocess_avalanche_report(self):
        self.create_directory(f'{self.layer_data_directory}/{self.working_region_directory}/report_data')

        self.create_microregion_definition()

        self.pull_avalanche_report()
        self.standardize_avalanche_report()

        avalanche_report_base_layer_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/avalanche_report_region_layer.tif'
        standardized_avalanche_report_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/standardized_avalanche_report.json'
        self.burn_geometries_in_raster(f'{self.layer_data_directory}/{self.working_region_directory}/height.tif', avalanche_report_base_layer_filepath, standardized_avalanche_report_filepath)
        self.create_danger_layer(avalanche_report_base_layer_filepath, standardized_avalanche_report_filepath)

    def create_microregion_definition(self):
        if self.working_region in [Region.BAVARIA, Region.TYROL]:
            microregions_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/microregions.json'
            self.pull_microregions(microregions_filepath, self.avalanche_report_microregions[self.working_region])
            microregions_converted_espg_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/microregions_converted_espg.json'
            self.convert_coordinate_reference(microregions_filepath, microregions_converted_espg_filepath)

    def pull_avalanche_report(self):
        avalanche_report_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/raw_avalanche_report'
        if self.working_region == Region.BAVARIA or self.working_region == Region.TYROL:
            self.download_avalanche_report_xml(f'{avalanche_report_filepath}.xml', self.avalanche_report_url[self.working_region])
        elif self.working_region == Region.SWITZERLAND:
            self.download_avalanche_report_json(f'{avalanche_report_filepath}.json', self.avalanche_report_url[self.working_region])

    def standardize_avalanche_report(self):
        avalanche_report_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/raw_avalanche_report'
        standardized_avalanche_report_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/standardized_avalanche_report.json'
        microregions_converted_espg_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/microregions_converted_espg.json'
        if self.working_region == Region.BAVARIA or self.working_region == Region.TYROL:
            self.create_standardized_avalanche_report_CAAMLV6(f'{avalanche_report_filepath}.xml', standardized_avalanche_report_filepath, microregions_converted_espg_filepath)
        elif self.working_region == Region.SWITZERLAND:
            standardized_avalanche_report_wrong_epsg_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/report_data/standardized_avalanche_report_wrong_epsg.json'
            self.create_standardized_avalanche_report_switzerland(f'{avalanche_report_filepath}.json', standardized_avalanche_report_wrong_epsg_filepath)
                
            self.convert_coordinate_reference(standardized_avalanche_report_wrong_epsg_filepath, standardized_avalanche_report_filepath)

    def download_avalanche_report_xml(self, file, url): # CHECK
        response = requests.get(url)
        if response.status_code == 200:
            data = response.content
            writeFile = open(file, 'wb')
            writeFile.write(data)
            writeFile.close()
            print('Downloaded avalanche report')
        else:
            print('Download of avalanche report doesn\'t work')
    
    def download_avalanche_report_json(self, file, url): # CHECK
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            writeFile = open(file, 'w+', encoding='utf-8')
            json.dump(data, writeFile, ensure_ascii=False, indent=4)
            writeFile.close()
            print('Downloaded avalanche report')
        else:
            print('Download of avalanche report doesn\'t work')
            
    def pull_microregions(self, file, regions): # CHECK TODO: umbauen das alle regionen untereinander gepackt werden TODO 
        # GitLab URL und Access Token (PAT)
        GITLAB_URL = "https://gitlab.com"  # oder eigene GitLab-Instanz
        PROJECT_ID = 25330421  # Die Project-ID deines Repos
        # GitLab API-Client initialisieren
        gl = gitlab.Gitlab(GITLAB_URL)

        # Projekt holen
        project = gl.projects.get(PROJECT_ID)

        for region in regions:
            
            FILE_PATH = f'public/micro-regions/{region}_micro-regions.geojson.json'  # Pfad der Datei im Repository
            BRANCH = "master"  # Oder ein anderer Branch
            # Datei herunterladen
            file_data = project.files.get(file_path=FILE_PATH, ref=BRANCH)

            if os.path.exists(file):
                data = file_data.decode()
                data = json.loads(data)
                new_features = data['features']
                with open(file, "r") as f:
                    f = json.load(f)
                    features = f['features']
                    for new_feature in new_features:
                        features.append(new_feature)
                    f['features'] = features
                with open(file, "w") as w:
                    json.dump(f, w, ensure_ascii=False, indent=4)
            else:
                with open(file, "wb") as f:
                    f.write(file_data.decode())

            file_data = None

    def convert_coordinate_reference(self, src_path, dst_path): # CHECK

        # Ziel SRS
        dst_srs = osr.SpatialReference()
        dst_srs = self.working_srs
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # Quell SRS
        driver = ogr.GetDriverByName('GeoJSON')
        src_file = driver.Open(src_path, 0)  # 0 bedeutet nur Lesen
        src_layer = src_file.GetLayer()
        src_srs = src_layer.GetSpatialRef()
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # Erstelle die Ausgabedatei (GeoJSON)
        dst_file = driver.CreateDataSource(dst_path)
        dst_layer = dst_file.CreateLayer('transformed_layer', srs=dst_srs)

        # Transformationseinrichtung
        coord_trans = osr.CoordinateTransformation(src_srs, dst_srs)

        # Kopiere Features und transformiere Geometrien
        for feature in src_layer:
            geom = feature.GetGeometryRef()
            geom.Transform(coord_trans)  # Transformation der Geometrie

            dst_layer.CreateFeature(feature)  # Speichern der transformierten Geometrie

        # Schließe die Datenquellen
        src_file = None
        dst_file = None

        print(f"Die GeoJSON-Datei wurde erfolgreich in das neue Koordinatensystem umgewandelt und unter {dst_path} gespeichert.")

    def burn_geometries_in_raster(self, src_raster_path, dst_raster_path, geojason_path): #CHECK

        driver = ogr.GetDriverByName('GeoJSON')
        geojason_file = driver.Open(geojason_path, 0)  # 0 bedeutet nur Lesen
        input_layer = geojason_file.GetLayer()

        # Erstelle ein OGR Memory-Dataset
        driver = ogr.GetDriverByName("Memory")
        data_source = driver.CreateDataSource("Memory")

        # Erstelle eine Layer für die Geometrien
        layer = data_source.CreateLayer("geometry_layer", srs= self.working_srs,geom_type=ogr.wkbPolygon)

        # Iteriere über die Features und füge die Geometrien zum Layer hinzu
        for feature in input_layer:
                geom = feature.GetGeometryRef()
                feature_def = layer.GetLayerDefn()
                ogr_feature = ogr.Feature(feature_def)
                ogr_feature.SetGeometry(geom)
                layer.CreateFeature(ogr_feature)

        src_file = gdal.Open(src_raster_path)

        rows, column = np.shape(src_file.ReadAsArray())

        driver = gdal.GetDriverByName('GTiff')
        mask_raster = driver.Create(dst_raster_path,
                                        int(column),
                                        int(rows),
                                        1,
                                        eType = gdal.GDT_Float32)
        mask_raster.SetProjection(self.working_projection)
        mask_raster.SetGeoTransform(self.working_geotransform)

        #gdal.RasterizeLayer(mask_raster, [1], layer, burn_values=[1], options=[""])
        for feature in layer:
            temp_ds = ogr.GetDriverByName("Memory").CreateDataSource("")  # Temporäre Datenquelle
            temp_layer = temp_ds.CreateLayer("temp_layer", layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)

            # Feature kopieren um ein einzelnes feature in raster zu brennen
            new_feature = feature.Clone()
            temp_layer.CreateFeature(new_feature)

            # Geometrie ins Raster brennen mit FID als Wert
            gdal.RasterizeLayer(mask_raster, [1], temp_layer, burn_values=[feature.GetFID()])

        mask_raster.FlushCache()  # Daten in die Datei schreiben
        mask_raster = None

        geojason_file = None
        src_file = None

    # (Grenzhöhe treeline abfangen)
    # 01 Lawinengefahr: [Gefahrenstufe, Ober-/Unterhalb, Grenzhöhe], ..
    # 02 Lawinenproblem: [Problemtyp, Ober-/Unterhalb, Grenzhöhe, Expositionen], ..
    # 03 Gefahrenmuster: [Mustertyp], ..
    # 04 Regionen

    def create_standardized_avalanche_report_CAAMLV6(self, src_file, dst_file, micro_region_file): # TODO patterns aufnehmen
        tag_prefix = '{http://caaml.org/Schemas/V6.0/Profiles/BulletinEAWS}'
        tree = ET.parse(src_file)
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
            file = open(micro_region_file, "r", encoding="utf-8")
            microregion_data = json.load(file)
            file.close()

            # self.layer_data_directory = 'layer_data'
            # self.working_region_directory = None
            
            

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
        
        file = open(micro_region_file, "r", encoding="utf-8")
        microregion_data = json.load(file)
        file.close()
        crs = microregion_data['crs']['properties']['name']
        # Create GeoJSON structure
        standardized_avalanche_report_dict = {"type": "FeatureCollection",
                                              "name": "standardized_avalanche_report",
                                              "crs": { "type": "name", "properties": { "name": crs } },
                                              "features": features}
    
        with open(dst_file, 'w') as f:
            json.dump(standardized_avalanche_report_dict, f, indent= 4)

    def create_standardized_avalanche_report_switzerland(self, src_path, dst_path):
        
        driver = ogr.GetDriverByName('GeoJSON')
        src_file = driver.Open(src_path, 0)  # 0 bedeutet nur Lesen
        src_layer = src_file.GetLayer()

        dst_file = driver.CreateDataSource(dst_path)
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

    def create_danger_layer(self, region_layer_path, avalanche_report_path):
        driver = ogr.GetDriverByName('GeoJSON')
        avalanche_report = driver.Open(avalanche_report_path, 0)
        avalanche_report_layers = avalanche_report.GetLayer()

        earlier_list = []
        later_list = []
        for feature in avalanche_report_layers:
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
        
        file = gdal.Open(region_layer_path)
        region_layer = np.array(file.ReadAsArray(), dtype=np.float32)
        region_layer.shape
        danger_layer_early = np.full(region_layer.shape, np.nan)
        danger_layer_late = np.full(region_layer.shape, np.nan)
        feature_count = avalanche_report_layers.GetFeatureCount()

        for i in range(feature_count):
            danger_layer_early = np.where(region_layer == np.float32(i), earlier_list[i], danger_layer_early)
        for i in range(feature_count):
            danger_layer_late = np.where(region_layer == np.float32(i), later_list[i], danger_layer_late)
        
        self.save_layer([Layers.DANGER_EARLY, Layers.DANGER_LATE], [danger_layer_early, danger_layer_late])

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
                if type(required_layertype) == Layers: # layertype != preprocessed data
                    self.calculateLayer(required_layertype)
                else: # layertype == preprocessed data
                    self.calculateLayer(type(required_layertype))
                file = gdal.Open(required_layer_filepath)
                array = np.array(file.ReadAsArray(), dtype=np.float32)
                layers_dict[required_layertype] = array
                file = None
        return layers_dict

    def save_layer(self, layertypes, layer_arrays):
        if type(layertypes) == Layers: # layertype != preprocessed data
            layertypes = [layertypes]
            layer_arrays = [layer_arrays]

        for i, layertype in enumerate(layertypes):
            layer_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/{layertype.value}'

            yLength, xLength = np.shape(layer_arrays[i])
            driver = gdal.GetDriverByName('GTiff')
            output_raster = driver.Create(layer_filepath,
                                          int(xLength),
                                          int(yLength),
                                          1,
                                          eType = gdal.GDT_Float32)

            output_raster.SetProjection(self.working_projection)
            output_raster.SetGeoTransform(self.working_geotransform)
            band = output_raster.GetRasterBand(1)
            band.SetNoDataValue(-9999)
            band.WriteArray(layer_arrays[i])          
            band.FlushCache()
            band.ComputeStatistics(False)

        #     rows, columns = np.shape(layer_array)
        #     driver = gdal.GetDriverByName('GTiff')
        #     output_raster = driver.Create(f'{self.map_folder}/{layertype.value}',
        #                                         int(columns),
        #                                         int(rows),
        #                                         1,
        #                                         eType = gdal.GDT_Float32)

        #     output_raster.SetProjection(self.projection)
        #     output_raster.SetGeoTransform(self.geotransform)
        #     band = output_raster.GetRasterBand(1)
        #     band.SetNoDataValue(-9999)
        #     band.WriteArray(layer_array)          
        #     band.FlushCache()
        #     band.ComputeStatistics(False)
        # else: # layertype == preprocessed data
        #     for i, l1 in enumerate(layertype):
        #         rows, columns = np.shape(layer_array[i])
        #         driver = gdal.GetDriverByName('GTiff')
        #         output_raster = driver.Create(f'{self.map_folder}/{l1.value}',
        #                                         int(columns),
        #                                         int(rows),
        #                                         1,
        #                                         eType = gdal.GDT_Float32)

        #         output_raster.SetProjection(self.projection)
        #         output_raster.SetGeoTransform(self.geotransform)
        #         band = output_raster.GetRasterBand(1)
        #         band.SetNoDataValue(-9999)
        #         band.WriteArray(layer_array[i])          
        #         band.FlushCache()
        #         band.ComputeStatistics(False)

##### SHOW #####

    def showLayer(self, layertypes):

        fig = pyplot.figure(figsize=(10, 8))

        for i, layertype in enumerate(layertypes):
            layer_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/{layertype.value}'

            ax = fig.add_subplot(1, len(layertypes), i+1)
            file = gdal.Open(layer_filepath)
            layer_array = file.ReadAsArray()
            ax.imshow(layer_array, cmap = 'viridis', extent=[660000, 670000, 5270000, 5280000])
            minn = layer_array.min()
            maxx = layer_array.max()
            norm = Normalize(minn, maxx)
            m = pyplot.cm.ScalarMappable(norm=norm, cmap= 'viridis')
            m.set_array([])
            pyplot.colorbar(m, ax=ax, orientation="horizontal")
        pyplot.show()

    def show3D(self, layertypes, cmaps, values, area):

        height_layer_filepath = f'{self.layer_data_directory}/{self.working_region_directory}/height.tif'

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
            file = gdal.Open(height_layer_filepath)
            height_array = file.ReadAsArray()
            Z = height_array[area[0]:area[0]+area[2], area[1]:area[1]+area[3]]
            ax.plot_surface(X, Y, Z, facecolors=fcolor)
            pyplot.colorbar(m, ax=ax, orientation="horizontal")
        pyplot.show()
