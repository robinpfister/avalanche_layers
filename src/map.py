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

from src.layers import Layers

class Region(Enum):
    BAVARIA = 'bavaria'
    SWITZERLAND = 'switzerland'
    TYROL = 'tyrol'

class Map():
    def __init__(self, layerCalculatorFactory):
        gdal.UseExceptions()

        self.layer_data_directory = 'layer_data'
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
        

##### PULL DGM #####

    def register_new(self):
        for region in Region:
            region_directory = region.value

            if os.path.isdir(f'input_data/DGM_{region_directory}'):
                data_available = bool(os.listdir(f'input_data/DGM_{region_directory}'))
            else:
                continue

            if data_available:
                os.makedirs(f'{self.layer_data_directory}/{region_directory}/preprocessed_data')
        
                self.merge_tif_files(f'input_data/DGM_{region_directory}', f'{self.layer_data_directory}/{region_directory}/height.tif')

                if region == Region.TYROL:
                    self.cut_1_edge(f'{self.layer_data_directory}/{region_directory}/height.tif')

                if region == Region.TYROL or region == Region.SWITZERLAND:
                    self.downscale_tif_file(f'{self.layer_data_directory}/{region_directory}/height.tif', 2)

    def cut_1_edge(self, path): # CHECK
        
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

    def downscale_tif_file(self, path, scale_factor): # CHECK
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

    def merge_tif_files(self, folder, output_file): # CHECK

        tif_files = glob.glob(f'{folder}/*.tif') 

        gdal.Warp(output_file, tif_files, format="GTiff", srcNodata=-9999, dstNodata=-9999)   

    def set_working_region(self, region):
        if isinstance(region, Region):
            self.working_region_directory = region.value

            # set working geotransform & projection
            file = gdal.Open(f'{self.layer_data_directory}/{self.working_region_directory}/height.tif')
            self.working_geotransform = tuple(file.GetGeoTransform()) #tuple(), str() nicht zwangsläufig nötig
            self.working_projection = str(file.GetProjection())
            self.working_srs = file.GetSpatialRef()

            file = None

##### PULL AVALANCHE REPORT #####

    def pull_new(self): # in einzelne funktionen für jede region umwandeln und in register_new() aufrufen TODO
        for region in Region:
            if region in [Region.BAVARIA, Region.TYROL]:
                self.set_working_region(region)
            region_directory = region.value

            #os.makedirs(f'{self.layer_data_directory}/{region_directory}/micro_regions')
            #os.makedirs(f'{self.layer_data_directory}/{region_directory}/avalanche_problem')
            os.makedirs(f'{self.layer_data_directory}/{region_directory}/report_data')

            # Pull Microregions
            if region in [Region.BAVARIA, Region.TYROL]:
                microregions_filepath = f'{self.layer_data_directory}/{region_directory}/report_data/microregions.json'
                self.pull_microregions(microregions_filepath, self.avalanche_report_microregions[region])
                microregions_converted_espg_filepath = f'{self.layer_data_directory}/{region_directory}/report_data/microregions_converted_espg.json'
                self.convert_coordinate_reference(microregions_filepath, microregions_converted_espg_filepath)

            # Pull Avalanche Report
            avalanche_report_filepath = f'{self.layer_data_directory}/{region_directory}/report_data/raw_avalanche_report'
            standardized_avalanche_report_filepath = f'{self.layer_data_directory}/{region_directory}/report_data/standardized_avalanche_report.json'
            if region == Region.BAVARIA or region == Region.TYROL:
                self.download_avalanche_report_xml(f'{avalanche_report_filepath}.xml', self.avalanche_report_url[region])
                self.create_standardized_avalanche_report_CAAMLV6(f'{avalanche_report_filepath}.xml', standardized_avalanche_report_filepath, microregions_converted_espg_filepath)
            else:
                self.download_avalanche_report_json(f'{avalanche_report_filepath}.json', self.avalanche_report_url[region])
                #if region == Region.TYROL:
                    #self.create_standardized_avalanche_report_tyrol(f'{avalanche_report_filepath}.json', standardized_avalanche_report_filepath)
                #elif region == Region.SWITZERLAND:
                    #self.create_standardized_avalanche_report_switzerland(f'{avalanche_report_filepath}.json', standardized_avalanche_report_filepath)
            
            if region == Region.BAVARIA or region == Region.TYROL:
                avalanche_report_base_layer_filepath = f'{self.layer_data_directory}/{region_directory}/report_data/avalanche_report_region_layer.tif'
                self.burn_geometries_in_raster(f'{self.layer_data_directory}/{region_directory}/height.tif', avalanche_report_base_layer_filepath, standardized_avalanche_report_filepath)
    
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

    def create_standardized_avalanche_report_CAAMLV6(self, src_file, dst_file, micro_region_file): # TODO
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

            self.layer_data_directory = 'layer_data'
            self.working_region_directory = None
            
            

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
                                              "name": "IT-32-TN_micro-regions",
                                              "crs": { "type": "name", "properties": { "name": crs } },
                                              "features": features}
    
        with open(dst_file, 'w') as f:
            json.dump(standardized_avalanche_report_dict, f, indent= 4)


        return standardized_avalanche_report_dict

    def create_standardized_avalanche_report_tyrol(self, src_file, dst_file): # TODO
        pass

    def create_standardized_avalanche_report_switzerland(self, src_file, dst_file): # TODO
        pass

##### LEGACY #####

    def createAvalancheReportLayersTYRL(self):

        regions = ["AT-07", 'IT-32-BZ', 'IT-32-TN']
        self.pull_avalanche_report(f'{self.map_folder}/avalanche_report/report.json', "https://static.avalanche.report/bulletins/latest/EUREGIO_de_CAAMLv6.json")
        self.pull_microregions(regions)
        for region in regions:
            self.convert_json_coordinate_reference(f'{self.map_folder}/micro_regions/{region}.json', f'{self.map_folder}/micro_regions/{region}_espg2056.json', 2056)
        self.create_avalanche_report_summary2(f'{self.map_folder}/avalanche_report/report.json', f'{self.map_folder}/avalanche_report/report_summary.json')
        self.create_geotif_report_features(regions, f'{self.map_folder}/avalanche_report/report_geometry.json')
        self.burn_geometries_in_raster(f'{self.map_folder}/{Layers.HEIGHT.value}', f'{self.map_folder}/{Layers.AVALANCHE_REPORT_MASK.value}', f'{self.map_folder}/avalanche_report/report_geometry.json')
        file = gdal.Open(f'{self.map_folder}/avalanche_report/avalanche_report_mask.tif')
        array = file.ReadAsArray()
        pyplot.imshow(array)
        pyplot.show()

    def create_avalanche_report_summary2(self, src_path, dst_path):

        file = open(src_path, "r", encoding="utf-8")
        data = json.load(file)
        file.close()

        avalanche_properties = []

        for feature in data['bulletins']:

            Risks = []

            for rrisk in feature['dangerRatings']:
                Risk = {'mainValue' : rrisk['mainValue'],
                        'upperBound' : rrisk['elevation']['upperBound'] if 'elevation' in rrisk and 'upperBound' in rrisk['elevation'] else None,
                        'lowerBound' : rrisk['elevation']['lowerBound'] if 'elevation' in rrisk and 'lowerBound' in rrisk['elevation'] else None}
                Risks.append(Risk)

            tendency = feature['tendency'][0]['tendencyType']
                
            Danger_Patterns = feature['customData']['LWD_Tyrol']['dangerPatterns']

            Avalanche_Problems = []

            for avalanche_problem in feature['avalancheProblems']:
                Avalanche_Problem = {'type' : avalanche_problem['problemType'],
                                     'upperBound' : avalanche_problem['elevation']['upperBound'] if 'upperBound' in avalanche_problem['elevation'] else None,
                                     'lowerBound' : avalanche_problem['elevation']['lowerBound'] if 'lowerBound' in avalanche_problem['elevation'] else None,
                                     'aspects' : avalanche_problem['aspects'],
                                     'snowpackStability' : avalanche_problem['snowpackStability'],
                                     'frequency' : avalanche_problem['frequency'],
                                     'avalancheSize' : avalanche_problem['avalancheSize']}
                Avalanche_Problems.append(Avalanche_Problem)
            
            Regions = []

            for rregion in feature['regions']:
                Regions.append(rregion['regionID'])

            avalanche_property = {'Risks' : Risks,
                                  'tendency' : tendency,
                                  'Danger_Patterns' : Danger_Patterns,
                                  'Avalanche_Problems' : Avalanche_Problems,
                                  'Regions' : Regions}
            
            avalanche_properties.append(avalanche_property)

        data = {'data' : avalanche_properties}     

        writeFile =open(dst_path, 'w', encoding='utf-8')
        json.dump(data, writeFile, ensure_ascii=False, indent=4)
        writeFile.close()

    def create_geotif_report_features(self, regions, dst_path):

# Die id's für 

            file = open(f'{self.map_folder}/avalanche_report/report_summary.json', "r", encoding="utf-8")
            summary_data = json.load(file)
            file.close()

            features = []

            for i, report in enumerate(summary_data['data']):

                polygones = []

                for region in regions:

                    file = open(f'{self.map_folder}/micro_regions/{region}_espg2056.json', "r", encoding="utf-8")
                    data = json.load(file)
                    file.close()

                    report_regions = report['Regions']
                    for report_region in report_regions:
                        if region in report_region:

                            for micro_region in data['features']:
                                if micro_region['properties']['id'] == report_region:
                                    for polygon in micro_region['geometry']['coordinates']:
                                        polygones.append(polygon)
                    
                feature = {'type' : 'Feature',
                           'properties' : {'id' : i},
                           'geometry' : {'type' : 'MultiPolygon',
                                         'coordinates' : polygones}}
                features.append(feature)
            
            data = {"type": "FeatureCollection",
                    "name": "IT-32-TN_micro-regions",
                    "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
                    "features": features}
            
            writeFile =open(dst_path, 'w', encoding='utf-8')
            json.dump(data, writeFile, ensure_ascii=False, indent=4)
            writeFile.close()
            
    def createAvalancheReportLayersCH(self):

        self.pull_avalanche_report(f'{self.map_folder}/avalanche_report/report.json', "https://aws.slf.ch/api/bulletin/caaml/v4/de/geojson")
        self.convert_json_coordinate_reference(f'{self.map_folder}/avalanche_report/report.json', f'{self.map_folder}/avalanche_report/report_espg2056.json', 2056)
        self.create_avalanche_report_summary(f'{self.map_folder}/avalanche_report/report_espg2056.json', f'{self.map_folder}/avalanche_report/report_summary.json')
        self.burn_geometries_in_raster(f'{self.map_folder}/{Layers.HEIGHT.value}', f'{self.map_folder}/{Layers.AVALANCHE_REPORT_MASK.value}', f'{self.map_folder}/avalanche_report/report_espg2056.json')
        self.create_avalanche_report_layers()

    def create_avalanche_report_summary(self, src_path, dst_path):

        driver = ogr.GetDriverByName('GeoJSON')
        src_file = driver.Open(src_path, 0)  # 0 bedeutet nur Lesen
        src_layer = src_file.GetLayer()

        avalanche_properties = []

        for feature in src_layer:

            Problems = []
            avalancheProblems = feature.GetField("avalancheProblems")
            avalancheProblems = json.loads(avalancheProblems)

            c_d = feature.GetField("customData")
            c_d = json.loads(c_d)
            agg = c_d['CH']['aggregation']

            for t in agg:
                if t['category'] == 'dry':
                    typ = t['problemTypes']


            for avalancheProblem in avalancheProblems:
                Problem = {}
                problemType = avalancheProblem["problemType"]
                if problemType not in typ:
                    continue
                dangerRatingValue = avalancheProblem["dangerRatingValue"]
                Problem['type'] = problemType
                Problem['mainLevel'] = dangerRatingValue
                Problem['subLevel'] = None
                Problem['elevationLowerBound'] = None
                Problem['elevationUpperBound'] = None
                Problem['aspects'] = None
                if "customData" in avalancheProblem:
                    customData = avalancheProblem["customData"]
                    if "CH" in customData:
                        ch = customData["CH"]
                        if "subdivision" in ch:
                            subLevel = ch["subdivision"]
                            Problem['subLevel'] = subLevel
                if "elevation" in avalancheProblem:
                    elevation = avalancheProblem["elevation"]
                    if "lowerBound" in elevation:
                        lowerBound = elevation["lowerBound"]
                        Problem['elevationLowerBound'] = lowerBound
                    if "upperBound" in elevation:
                        upperBound = elevation["upperBound"]
                        Problem['elevationUpperBound'] = upperBound
                if "aspects" in avalancheProblem:
                    aspects = avalancheProblem["aspects"]
                    Problem['aspects'] = aspects
                Problems.append(Problem)
            avalanche_properties.append(Problems)
            data = {'data' : avalanche_properties}

            writeFile =open(dst_path, 'w', encoding='utf-8')
            json.dump(data, writeFile, ensure_ascii=False, indent=4)
            writeFile.close()

    def create_avalanche_report_layers(self):

        file = open(f'{self.map_folder}/avalanche_report/report_summary.json', 'r')
        data = json.load(file)['data']
        file = None

        file_height = gdal.Open(f'{self.map_folder}/{Layers.HEIGHT.value}')
        height = np.array(file_height.ReadAsArray(), dtype=np.float32)
        file_height = None

        file_dirlvl = gdal.Open(f'{self.map_folder}/{Layers.DIRECTION_LEVEL.value}')
        dirlvl = np.array(file_dirlvl.ReadAsArray(), dtype=np.float32)
        file_dirlvl = None

        file = gdal.Open(f'{self.map_folder}/{Layers.AVALANCHE_REPORT_MASK.value}')
        mask_array = np.array(file.ReadAsArray(), dtype=np.int8)
        rows, columns = np.shape(mask_array)

        wind_slab = np.zeros((rows, columns), dtype=np.float32)
        new_snow = np.zeros((rows, columns), dtype=np.float32)
        persistent_weak_layer = np.zeros((rows, columns), dtype=np.float32)
        no_distinct_problem = np.zeros((rows, columns), dtype=np.float32)

        for i, problems in enumerate(data):
            for problem in problems:

                if problem['mainLevel'] == 'low':
                    level = 1
                elif problem['mainLevel'] == 'moderate':
                    level = 3
                elif problem['mainLevel'] == 'erheblich':
                    level = 6
                elif problem['mainLevel'] == 'groß':
                    level = 9
                elif problem['mainLevel'] == 'sehr_groß':
                    level = 12

                if problem['subLevel'] == 'plus':
                    level = level + 1
                elif problem['subLevel'] == 'minus':
                    level = level - 1

                dir_array = np.zeros((np.shape(new_snow)), dtype=np.float32)
                
                directions = problem['aspects']
                if directions != None:
                    for direction in directions:
                        if direction == 'N':
                            direction = 3
                        elif direction == 'NE':
                            direction = 4
                        elif direction == 'E':
                            direction = 5
                        elif direction == 'SE':
                            direction = 6
                        elif direction == 'S':
                            direction = 7
                        elif direction == 'SW':
                            direction = 8
                        elif direction == 'W':
                            direction = 1
                        elif direction == 'NW':
                            direction = 2
                        if direction == 0:
                            print('Fehler')
                        dir_array = np.where(dirlvl == direction, 1, dir_array)
                
                test = np.zeros((np.shape(new_snow)), dtype=np.float32)

                lowerBound = problem['elevationLowerBound']
                if lowerBound != None:
                    lowerBound = np.float32(lowerBound)
                    test = np.where(height > lowerBound, 1, 0)
                upperBound = problem['elevationUpperBound']
                if upperBound != None:
                    upperBound = np.float32(upperBound)
                    test = np.where(height < upperBound, test, 0)

                test = np.where(dir_array == 1, test, 0)
                test = np.where(test == 1, np.float32(level), 0)

                if problem['type'] == 'wind_slab':
                    wind_slab = np.where(mask_array == i, test, wind_slab)

                elif problem['type'] == 'new_snow':
                    new_snow = np.where(mask_array == i, test, wind_slab)

                elif problem['type'] == 'persistent_weak_layer':
                    persistent_weak_layer = np.where(mask_array == i, test, wind_slab)

                elif problem['type'] == 'no_distinct_problem':
                    no_distinct_problem = np.where(mask_array == i, test, wind_slab)

        lt = [Layers.AVALANCHE_PROBLEM.WIND_SLAB, Layers.AVALANCHE_PROBLEM.NEW_SNOW, Layers.AVALANCHE_PROBLEM.PERSISTENT_WEAK_LAYER, Layers.AVALANCHE_PROBLEM.NO_DISTINCT_PROBLEM]
        l = [wind_slab, new_snow, persistent_weak_layer, no_distinct_problem]

        for lt1, l1 in zip(lt, l):
            driver = gdal.GetDriverByName('GTiff')
            output_raster = driver.Create(f'{self.map_folder}/{lt1.value}',
                                            int(columns),
                                            int(rows),
                                            1,
                                            eType = gdal.GDT_Float32)

            output_raster.SetProjection(self.projection)
            output_raster.SetGeoTransform(self.geotransform)
            band = output_raster.GetRasterBand(1)
            band.SetNoDataValue(-9999)
            band.WriteArray(l1)          
            band.FlushCache()
            band.ComputeStatistics(False)

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
