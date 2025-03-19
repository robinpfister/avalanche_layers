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

from src.layers import Layers


class Map():
    def __init__(self, layerCalculatorFactory):
        gdal.UseExceptions()

        self.layerCalculatorFactory = layerCalculatorFactory

        self.map_folder = None
 
        self.projection = None
        self.geotransform = None

    def register_new_map(self, dgm_folder, map_folder, tirol):
        
        self.map_folder = map_folder
        os.makedirs(f'{self.map_folder}/preprocessed_data')
        os.makedirs(f'{self.map_folder}/avalanche_problem')
        os.makedirs(f'{self.map_folder}/avalanche_report')
        os.makedirs(f'{self.map_folder}/micro_regions')
        self.merge_tif_files(dgm_folder, f'{self.map_folder}/height.tif')

        if tirol == True:
            self.cut_1_edge(f'{self.map_folder}/height.tif')

        #self.downscale_tif_file(f'{self.map_folder}/height.tif', 2)

        

        file = gdal.Open(f'{self.map_folder}/height.tif')
        self.geotransform = tuple(file.GetGeoTransform()) #tuple(), str() nicht zwangsläufig nötig
        self.projection = str(file.GetProjection())
        file = None

    def cut_1_edge(self, path):
        
        file = gdal.Open(path)
        layer_array = file.ReadAsArray()
        geotransform = file.GetGeoTransform()
        projektion = file.GetProjection()
        file = None
        new_array = np.copy(layer_array[0:2000, 0:2500])
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

    def downscale_tif_file(self, path, scale_factor): # private
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
        print(rows)
        print(columns)

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

    def merge_tif_files(self, folder, output_file): # private

        tif_files = glob.glob(f'{folder}/*.tif') 

        gdal.Warp(output_file, tif_files, format="GTiff", srcNodata=-9999, dstNodata=-9999)   

    def register_existing_map(self, map_folder):

        self.map_folder = map_folder

        file = gdal.Open(f'{self.map_folder}/height.tif')
        self.geotransform = tuple(file.GetGeoTransform()) #tuple(), str() nicht zwangsläufig nötig
        self.projection = str(file.GetProjection())
        file = None

    def calculateLayer(self, layertype):
        layerCalculator = self.layerCalculatorFactory.get_calculator(layertype)
        layers = {}
        for l in layerCalculator.required_layers:
            if os.path.isfile(f'{self.map_folder}/{l.value}'):
                file = gdal.Open(f'{self.map_folder}/{l.value}')
                array = np.array(file.ReadAsArray(), dtype=np.float32)
                layers[l] = array
                file = None
            else:
                if type(l) == Layers:
                    self.calculateLayer(l)
                    file = gdal.Open(f'{self.map_folder}/{l.value}')
                    array = np.array(file.ReadAsArray(), dtype=np.float32)
                    layers[l] = array
                    file = None
                else:
                    self.calculateLayer(type(l))
                    file = gdal.Open(f'{self.map_folder}/{l.value}')
                    array = np.array(file.ReadAsArray(), dtype=np.float32)
                    layers[l] = array
                    file = None
        start = time.time()
        calculated_layers = layerCalculator.calculate(layers)
        ende = time.time()
        print(f'[LOG] Berechnungzeit {layertype}: {round(ende-start, 2)}s')
        if type(layertype) == Layers:
            rows, columns = np.shape(calculated_layers)
            driver = gdal.GetDriverByName('GTiff')
            output_raster = driver.Create(f'{self.map_folder}/{layertype.value}',
                                                int(columns),
                                                int(rows),
                                                1,
                                                eType = gdal.GDT_Float32)

            output_raster.SetProjection(self.projection)
            output_raster.SetGeoTransform(self.geotransform)
            band = output_raster.GetRasterBand(1)
            band.SetNoDataValue(-9999)
            band.WriteArray(calculated_layers)          
            band.FlushCache()
            band.ComputeStatistics(False)
            # Einzelnes Speichern
        else:
            for i, l1 in enumerate(layertype):
                rows, columns = np.shape(calculated_layers[i])
                driver = gdal.GetDriverByName('GTiff')
                output_raster = driver.Create(f'{self.map_folder}/{l1.value}',
                                                int(columns),
                                                int(rows),
                                                1,
                                                eType = gdal.GDT_Float32)

                output_raster.SetProjection(self.projection)
                output_raster.SetGeoTransform(self.geotransform)
                band = output_raster.GetRasterBand(1)
                band.SetNoDataValue(-9999)
                band.WriteArray(calculated_layers[i])          
                band.FlushCache()
                band.ComputeStatistics(False)
    
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

    def pull_microregions(self, regions):

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

            # Datei speichern
            with open(f'{self.map_folder}/micro_regions/{region}.json', "w", encoding="utf-8") as f:
                f.write(file_data.decode().decode("utf-8"))

            file_data = None

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

    def pull_avalanche_report(self, file, url):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            writeFile = open(file, 'w+', encoding='utf-8')
            json.dump(data, writeFile, ensure_ascii=False, indent=4)
            writeFile.close()
            print('Downloaded avalanche report')
        else:
            print('Download of avalanche report doesn\'t work')

    def convert_json_coordinate_reference(self, src_path, dst_path, espg):
        # Ziel-Koordinatensystem (ETRS89 - EPSG:4258)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(espg)

        # Öffne die Eingabedatei
        driver = ogr.GetDriverByName('GeoJSON')
        src_file = driver.Open(src_path, 0)  # 0 bedeutet nur Lesen
        src_layer = src_file.GetLayer()

        # Hole das Quell-Koordinatensystem
        src_srs = src_layer.GetSpatialRef()

        # Erstelle die Ausgabedatei (GeoJSON)
        dst_file = driver.CreateDataSource(dst_path)
        dst_layer = dst_file.CreateLayer('transformed_layer', srs=src_srs)

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

    def burn_geometries_in_raster(self, src_raster_path, dst_raster_path, geojason_path):

        driver = ogr.GetDriverByName('GeoJSON')
        geojason_file = driver.Open(geojason_path, 0)  # 0 bedeutet nur Lesen
        input_layer = geojason_file.GetLayer()

        # Erstelle ein OGR Memory-Dataset
        driver = ogr.GetDriverByName("Memory")
        data_source = driver.CreateDataSource("Memory")

        # Erstelle ein Spatial Reference Objekt für LV95
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(2056)

        # Erstelle eine Layer für die Geometrien
        layer = data_source.CreateLayer("geometry_layer", srs= srs,geom_type=ogr.wkbPolygon)

        # Iteriere über die Features und füge die Geometrien zum Layer hinzu
        for feature in input_layer:
                geom = feature.GetGeometryRef()
                feature_def = layer.GetLayerDefn()
                ogr_feature = ogr.Feature(feature_def)
                ogr_feature.SetGeometry(geom)
                layer.CreateFeature(ogr_feature)

        # # DEBUG
        # print(f"Anzahl der Geometrien: {layer.GetFeatureCount()}")

        src_file = gdal.Open(src_raster_path)

        rows, column = np.shape(src_file.ReadAsArray())

        driver = gdal.GetDriverByName('GTiff')
        mask_raster = driver.Create(dst_raster_path,
                                        int(column),
                                        int(rows),
                                        1,
                                        eType = gdal.GDT_Float32)
        mask_raster.SetProjection(self.projection)
        mask_raster.SetGeoTransform(self.geotransform)

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

    def showLayer(self, layertypes):

        fig = pyplot.figure(figsize=(10, 8))

        for i, layertype in enumerate(layertypes):
            ax = fig.add_subplot(1, len(layertypes), i+1)
            file = gdal.Open(f'{self.map_folder}/{layertype.value}')
            layer_array = file.ReadAsArray()
            ax.imshow(layer_array, cmap = 'viridis', extent=[660000, 670000, 5270000, 5280000])
            minn = layer_array.min()
            maxx = layer_array.max()
            norm = Normalize(minn, maxx)
            m = pyplot.cm.ScalarMappable(norm=norm, cmap= 'viridis')
            m.set_array([])
            pyplot.colorbar(m, ax=ax, orientation="horizontal")
        pyplot.show()

    def show3D(self, layertypes, heightmaps, cmaps, values, area):

        fig = pyplot.figure(figsize=(10, 8))

        x = np.linspace(0, area[3]-1, area[3])
        y = np.linspace(0, area[2]-1, area[2])
        X,Y = np.meshgrid(x, y)

        for i, layertype in enumerate(layertypes):
                    file = gdal.Open(f'{self.map_folder}/{layertype.value}')
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
                    file = gdal.Open(f'{self.map_folder}/{heightmaps[i].value}')
                    height_array = file.ReadAsArray()
                    Z = height_array[area[0]:area[0]+area[2], area[1]:area[1]+area[3]]
                    ax.plot_surface(X, Y, Z, facecolors=fcolor)
                    pyplot.colorbar(m, ax=ax, orientation="horizontal")
        pyplot.show()