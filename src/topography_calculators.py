# internal imports
from layer import Layer
from layer_calculator_factory import LayerCalculator

# external imports
import numpy as np
import numba as nb

class HeightLevelCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.HEIGHT]
    
    def calculate(self, layers):
         height = layers.get(Layer.HEIGHT)

         height_level = (height // 100) * 100

         return height_level

class AspectCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.COEFFICIENT.D, Layer.COEFFICIENT.E]
        
    def calculate(self, layers):
        D = layers.get(Layer.COEFFICIENT.D)
        E = layers.get(Layer.COEFFICIENT.E)
            
        aspect = np.arctan2(-E, -D) 
        aspect = (aspect * (180 / np.pi)) + 180

        return aspect

class AspectLevelCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.ASPECT]
    
    def calculate(self, layers):
        aspect = layers.get(Layer.ASPECT)

        aspect_level = np.where(aspect > 22.5, 2, 1)
        aspect_level = np.where(aspect > 67.5, 3, aspect_level)
        aspect_level = np.where(aspect > 112.5, 4, aspect_level)
        aspect_level = np.where(aspect > 157.5, 5, aspect_level)
        aspect_level = np.where(aspect > 202.5, 6, aspect_level)
        aspect_level = np.where(aspect > 247.5, 7, aspect_level)
        aspect_level = np.where(aspect > 292.5, 8, aspect_level)
        aspect_level = np.where(aspect > 337.5, 1, aspect_level)
        
        return aspect_level
        
class SlopeCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.COEFFICIENT.D, Layer.COEFFICIENT.E]
    
    def calculate(self, layers):
        D = layers.get(Layer.COEFFICIENT.D)
        E = layers.get(Layer.COEFFICIENT.E)
        
        slope_vector = ((D ** 2) + (E ** 2)) ** (1/2)
        slope_angle = np.arctan((slope_vector)) * (180 / np.pi)
        
        return slope_angle

class SlopeLevelCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.SLOPE]
    
    def calculate(self, layers):
        slope = layers.get(Layer.SLOPE)

        slope_level = np.where(slope > 20, 1, 0)
        slope_level = np.where(slope > 25, 2, slope_level)
        slope_level = np.where(slope > 30, 3, slope_level)
        slope_level = np.where(slope > 35, 4, slope_level)
        slope_level = np.where(slope > 40, 5, slope_level)
        slope_level = np.where(slope > 45, 6, slope_level)
        slope_level = np.where(slope > 50, 7, slope_level)
        slope_level = np.where(slope > 55, 8, slope_level)
        slope_level = np.where(slope > 60, 9, slope_level)

        return slope_level

class ProfileCurvatureCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.COEFFICIENT.A, 
                                Layer.COEFFICIENT.B, 
                                Layer.COEFFICIENT.C, 
                                Layer.COEFFICIENT.D, 
                                Layer.COEFFICIENT.E]
        self.z_factor = 10
    
    def calculate(self, layers):
        A = layers.get(Layer.COEFFICIENT.A)
        B = layers.get(Layer.COEFFICIENT.B)
        C = layers.get(Layer.COEFFICIENT.C)
        D = layers.get(Layer.COEFFICIENT.D)
        E = layers.get(Layer.COEFFICIENT.E)

        denominator = ((D ** 2) + (E ** 2)) * ((1 + (D ** 2) + (E ** 2)) ** (3/2))
        denominator = np.where(denominator == 0, np.nan, denominator)
        profile_curvature = ((-2 * ((A * (D ** 2)) + (B * (E ** 2)) + (C * D * E))) / denominator) * (100 / self.z_factor)

        return profile_curvature

class PlanCurvatureCalculator(LayerCalculator):
    def __init__(self):
        self.required_layers = [Layer.COEFFICIENT.A, 
                                Layer.COEFFICIENT.B, 
                                Layer.COEFFICIENT.C, 
                                Layer.COEFFICIENT.D, 
                                Layer.COEFFICIENT.E]
        self.z_factor = 10
    
    def calculate(self, layers):
        A = layers.get(Layer.COEFFICIENT.A)
        B = layers.get(Layer.COEFFICIENT.B)
        C = layers.get(Layer.COEFFICIENT.C)
        D = layers.get(Layer.COEFFICIENT.D)
        E = layers.get(Layer.COEFFICIENT.E)

        denominator = (((D ** 2) + (E ** 2)) ** (3/2))
        denominator = np.where(denominator == 0, np.nan, denominator)
        plan_curvature = ((-2 * ((B * (D ** 2)) + (A * (E ** 2)) - (C * D * E))) / denominator) * (100 / self.z_factor)

        return plan_curvature

class ShapeCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.PROFILE_CURVATURE, Layer.PLAN_CURVATURE]
    
    def calculate(self, layers):
        profile_curvature = layers.get(Layer.PROFILE_CURVATURE)
        plan_curvature = layers.get(Layer.PLAN_CURVATURE)

        shape = np.where(profile_curvature < -0.2, 1, 2)
        shape = np.where(profile_curvature > 0.2, 3, shape)
        shape = np.where(plan_curvature < -0.2, shape, shape+3)
        shape = np.where(plan_curvature > 0.2, shape+3, shape)
        
        #check nan-values in input data
        shape = np.where(np.isnan(profile_curvature), np.nan, shape)
        shape = np.where(np.isnan(plan_curvature), np.nan, shape)

        return shape

class FoldCloseCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.ASPECT]
        
    def calculate(self, layers):
        aspect = layers.get(Layer.ASPECT)

        fold = self.get_highest_difference(aspect)

        return fold
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def get_highest_difference(data):
        y_length, x_length = data.shape
        highest_difference = np.zeros((y_length, x_length))

        for y in nb.prange(1, y_length - 1):
            for x in range(1, x_length - 1):
                aspect_around = data[y - 1 : y + 2, x - 1 : x + 2]
                difference = np.absolute(aspect_around - data[y, x])
                difference = np.where(difference > 180, 360 - difference, difference)
                highest_difference[y, x] = difference.max()
        
        return highest_difference

class FoldFarCalculator(LayerCalculator):
    def __init__(self):
        self.required_layers = [Layer.ASPECT]
        
    def calculate(self, layers):
        aspect = layers.get(Layer.ASPECT)

        fold = self.get_highest_difference(aspect)

        return fold
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def get_highest_difference(data):
        y_length, x_length = data.shape
        highest_difference = np.zeros((y_length, x_length)).astype(np.float32)

        def get_projected_point(radius, alpha_rad, y, x):
            y = y + (radius * np.sin(alpha_rad))
            x = x + (radius * np.cos(alpha_rad))
            return y, x

        rad_list = [np.pi * (0) / 180,
                    np.pi * (36) / 180,
                    np.pi * (72) / 180,
                    np.pi * (108) / 180,
                    np.pi * (144) / 180,
                    np.pi * (180) / 180,
                    np.pi * (216) / 180,
                    np.pi * (252) / 180,
                    np.pi * (288) / 180,
                    np.pi * (324) / 180]

        indice_y = np.zeros(10).astype(np.int8)
        indice_x = np.zeros(10).astype(np.int8)

        for i, rad in enumerate(rad_list):
            y, x = get_projected_point(10, rad, 0, 0)
            indice_y[i] = int(y)
            indice_x[i] = int(x)

        for y in nb.prange(10, y_length - 10):
            for x in range(10, x_length - 10):
                aspect_around = np.zeros(10).astype(np.float32)
                for i in range(10):
                    aspect_around[i] = data[y+indice_y[i], x+indice_x[i]]
                difference = np.absolute(aspect_around - data[y, x])
                difference = np.where(difference > 180, 360 - difference, difference)
                highest_difference[y, x] = difference.max()
        return highest_difference

class RelevantSlopeAreaCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.SLOPE, Layer.PLAN_CURVATURE]
        self.r_max = 40
    
    def calculate(self, layers):
        plan_curvature = layers.get(Layer.PLAN_CURVATURE)
        slope = layers.get(Layer.SLOPE)

        form_mask = np.where(np.absolute(plan_curvature) > 0.2, 1, 0)

        alpha_a, r1_a, r2_a = self.calculate_a_form(plan_curvature)
        alpha_d, r1_d, r2_d = self.calculate_d_form(plan_curvature, slope)

        alpha = np.where(form_mask == 1, alpha_a, alpha_d)
        r1 = np.where(form_mask == 1, r1_a, r1_d)
        r2 = np.where(form_mask == 1, r2_a, r2_d)

        return alpha, r1, r2, form_mask
    
    def calculate_rs_1(self, plan_curvature):
        c_high = -0.35
        c_medium = -0.2
        c_low = -0.04

        rs_high = 0
        rs_medium = 0.25
        rs_low = 1

        rs_1 = np.where(plan_curvature > c_high, (rs_medium * ((plan_curvature - c_high)/(c_medium - c_high))), rs_high)
        rs_1 = np.where(plan_curvature > c_medium, (rs_medium + (((rs_low - rs_medium) * (plan_curvature - c_medium)) / (c_low - c_medium))), rs_1)
        rs_1 = np.where(plan_curvature >= c_low, rs_low, rs_1)

        return rs_1

    def calculate_a_form(self, plan_curvature):
        rs_1 = self.calculate_rs_1(plan_curvature)
        rs_2 = self.calculate_rs_2a(plan_curvature)

        r1 = self.r_max * rs_1 * rs_2
        r2 = r1
        alpha = np.full(plan_curvature.shape, 180)

        return alpha, r1, r2

    def calculate_rs_2a(self, plan_curvature):
        c_high = 0.5
        c_low = 0.2

        rs_high = 1.5
        rs_low = 1

        rs_2 = np.where(plan_curvature > c_low, (rs_low + (((plan_curvature - c_low) * (rs_high - rs_low)) / (c_high - c_low))), rs_low)
        rs_2 = np.where(plan_curvature >= c_high, rs_high, rs_2)

        return rs_2

    def calculate_d_form(self, plan_curvature, slope):
        alpha = self.calculate_alpha(slope)

        rs_1 = self.calculate_rs_1(plan_curvature)
        rs_2 = self.calculate_rs_2d(slope)
        r_relative = self.calculate_r_relative(slope)
        t_5 = self.calculate_t_5(alpha, r_relative)

        r_1 = rs_1 * rs_2 * np.sqrt((np.pi * (self.r_max ** 2)) / (2 * t_5))
        r_2 = r_1 * r_relative

        return alpha, r_1, r_2

    def calculate_alpha(self, slope):
        alpha_high = 30
        alpha_low = 120

        s_high = 45
        s_low = 5

        #slope = np.where(np.isnan(slope), 0, slope)
        alpha = np.where(slope > s_low, alpha_low - (((alpha_low - alpha_high) * (slope - s_low)) / (s_high - s_low)), np.nan)
        alpha = np.where(slope >= s_high, alpha_high, alpha)

        return alpha

    def calculate_rs_2d(self, slope):
        s_high = 55
        s_low = 25

        rs_high = 2.5
        rs_low = 1

        rs_2 = np.where(slope > s_low, rs_low + (((slope - s_low) * (rs_high - rs_low)) / (s_high - s_low)), rs_low)
        rs_2 = np.where(slope >= s_high, rs_high, rs_2)

        return rs_2
    
    def calculate_r_relative(self, slope):
        s_high = 45
        s_low = 5

        r_high = 0.1
        r_low = 0.3

        r_relative = np.where(slope > s_low, r_low - (((r_low - r_high) * (slope - s_low)) / (s_high - s_low)), np.nan)
        r_relative = np.where(slope >= s_high, r_high, r_relative)

        return r_relative

    def calculate_t_5(self, alpha, r_relative):
        a_rad = alpha * (np.pi / 180)
        t_1 = (a_rad / 4) - (((np.sin((a_rad / 2))) * (np.cos(a_rad / 2))) / 2)
        t_2 = (r_relative + 1) * np.cos(a_rad / 2)
        t_3 = r_relative * np.sin(a_rad / 2)
        t_4 = ((np.sin(a_rad / 2)) * (1 - r_relative)) / 2
        t_5 = (t_1 * (1 + (r_relative ** 2))) + (t_2 * (t_3 + t_4))

        return t_5

class RsaMaxSlopeCalculator(LayerCalculator):

    def __init__(self):
        self.required_layers = [Layer.RELEVANT_SLOPE_AREA.ALPHA, Layer.RELEVANT_SLOPE_AREA.R1, Layer.RELEVANT_SLOPE_AREA.R2, Layer.RELEVANT_SLOPE_AREA.FORM, Layer.ASPECT, Layer.SLOPE, Layer.FOLD]
    
    def calculate(self, layers):
        alpha = layers.get(Layer.RELEVANT_SLOPE_AREA.ALPHA)
        r1 = layers.get(Layer.RELEVANT_SLOPE_AREA.R1)
        r2 = layers.get(Layer.RELEVANT_SLOPE_AREA.R2)
        mask = layers.get(Layer.RELEVANT_SLOPE_AREA.FORM)
        aspect = layers.get(Layer.ASPECT)
        slope = layers.get(Layer.SLOPE)
        fold = layers.get(Layer.FOLD)

        max_slope = self.calculate_max_slope(alpha, r1, r2, mask, aspect, slope, fold)

        return max_slope

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def calculate_max_slope(alpha, r1, r2, mask, aspect, slope, fold):

        def rasterize_polygon(poly_y_w, poly_x, fold, slope, y_1, x_1):
            min_x = int(np.floor(np.min(poly_x)))
            max_x = int(np.ceil(np.max(poly_x)))
            min_y = int(np.floor(np.min(poly_y_w)))
            max_y = int(np.ceil(np.max(poly_y_w)))
   
            #check if RSA geometry is inside the data layer -> if not return zero values
            layer_size_y, layer_size_x = fold.shape
            if min_y <= 0 or min_x <= 0:
                return 0 # auf np.nan unstellen
            elif max_y >= layer_size_y or max_x >= layer_size_x:
                return 0
               
            # add extra row/column at each side -> to be safe that the RSA-Geometry is in the bounding box
            bounding_box_x_length = max_x - min_x + 2
            bounding_box_y_length = max_y - min_y + 2
            #print(bounding_box_x_length)
            #print(bounding_box_y_length)
                    
            mask = np.zeros((bounding_box_y_length, bounding_box_x_length))
            #print(mask)

            yy, xx = np.indices((bounding_box_y_length, bounding_box_x_length)) # x/y indice array
            # offset for indice array -> indices like in data layer
            yy = yy + min_y - 1 
            xx = xx + min_x - 1
            
            for i in range(bounding_box_y_length):
                for j in range(bounding_box_x_length):
                    x = xx[i, j]
                    y = yy[i, j]
                    if (x >= min_x and x <= max_x and y >= min_y and y <= max_y):
                        n = len(poly_x)
                        inside = False
                        k = n - 1
                        for l in range(n):
                            if ((poly_y_w[l] > y) != (poly_y_w[k] > y)) and \
                            (x < (poly_x[k] - poly_x[l]) * (y - poly_y_w[l]) / (poly_y_w[k] - poly_y_w[l]) + poly_x[l]):
                                inside = not inside
                            k = l
                        if inside:
                            fy = int(min_y - 1 + i)
                            fx = int(min_x - 1 + j)
                            if 0 <= fy < fold.shape[0] and 0 <= fx < fold.shape[1]:
                                if fold[fy, fx] > 45:
                                    mask[i, j] = 2
                                else:
                                    mask[i, j] = 1

            center_y = int(y_1 - min_y + 1)
            center_x = int(x_1 - min_x + 1)  
            if 0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1]:         
                mask[center_y, center_x] = 5
                                            
            for i in range(bounding_box_y_length):
                for j in range(bounding_box_x_length):
                    if mask[i, j] == 2:
                                    
                        dir_y = i - center_y
                        dir_x = j - center_x
                        angle = np.arctan2(dir_y, dir_x)
                                
                        dir_y = np.sin(angle)
                        dir_x = np.cos(angle)
                        for z in range(100):
                            iter = z+1
                            y_w = int((i+(iter*dir_y)))
                            x_w = int((j+(iter*dir_x)))
                            if bounding_box_y_length-2 > y_w > 2 and bounding_box_x_length-2 >  x_w > 2:
                                if mask[y_w+2, x_w] == 1:
                                    mask[y_w+2, x_w] = 3
                                if mask[y_w-2, x_w] == 1:
                                    mask[y_w-2, x_w] = 3
                                if mask[y_w, x_w+2] == 1:
                                    mask[y_w, x_w+2] = 3
                                if mask[y_w, x_w-2] == 1:
                                    mask[y_w, x_w-2] = 3
                            if bounding_box_y_length-1 > y_w > 1 and bounding_box_x_length-1 >  x_w > 1:
                                if mask[y_w, x_w] == 1:
                                    mask[y_w, x_w] = 3
                                if mask[y_w, x_w+1] == 1:
                                    mask[y_w, x_w+1] = 3
                                if mask[y_w-1, x_w] == 1:
                                    mask[y_w-1, x_w] = 3
                                if mask[y_w-1, x_w-1] == 1:
                                    mask[y_w-1, x_w-1] = 3
                                if mask[y_w-1, x_w+1] == 1:
                                    mask[y_w-1, x_w+1] = 3
                                if mask[y_w+1, x_w] == 1:
                                    mask[y_w+1, x_w] = 3
                                if mask[y_w+1, x_w-1] == 1:
                                    mask[y_w+1, x_w-1] = 3
                                if mask[y_w+1, x_w+1] == 1:
                                    mask[y_w+1, x_w+1] = 3
                            else:
                                break
            mask = np.where(mask != 1, 0, 1)

            slope_mask = np.where(mask == 1, slope[min_y-1: min_y-1 + bounding_box_y_length, min_x-1 : min_x-1 + bounding_box_x_length], 0)
            max_slope = slope_mask.max()

            return max_slope
 
        def get_projected_point(radius, alpha_rad, y_l, x_l):
            y_l = y_l + (radius * np.sin(alpha_rad))
            x_l = x_l + (radius * np.cos(alpha_rad))
            return y_l, x_l

        max_slope = np.zeros(r1.shape)

        for y in nb.prange(r1.shape[0]):
            for x in range(r1.shape[0]):

                if np.isnan(r1[y, x]) or r1[y, x] == 0: #
                    continue

                poly_y_g = np.zeros(10, dtype=np.float64)
                poly_x_g = np.zeros(10, dtype=np.float64)
                
                # create angles in degree for polygon corner points
                if mask[y, x]: # A-Form
                    r1_start_radian = np.pi * (0) / 180
                    r1_mid1_radian = np.pi * (36) / 180
                    r1_mid2_radian = np.pi * (72) / 180
                    r1_mid3_radian = np.pi * (108) / 180
                    r1_end_radian = np.pi * (144) / 180
                    r2_start_radian = np.pi * (180) / 180
                    r2_mid1_radian = np.pi * (216) / 180
                    r2_mid2_radian = np.pi * (252) / 180
                    r2_mid3_radian = np.pi * (288) / 180
                    r2_end_radian = np.pi * (324) / 180
                else: # D-Form
                    r1_start_radian = np.pi * ((180+aspect[y, x] - alpha[y, x]/2) / 180)
                    r1_mid1_radian = np.pi * ((180+aspect[y, x] - alpha[y, x]/4) / 180)
                    r1_mid2_radian = np.pi * ((180+aspect[y, x]) / 180)
                    r1_mid3_radian = np.pi * ((180+aspect[y, x] + alpha[y, x]/4) / 180)
                    r1_end_radian = np.pi * ((180+aspect[y, x] + alpha[y, x]/2) / 180)
                    r2_start_radian = np.pi * ((aspect[y, x] - alpha[y, x]/2) / 180)
                    r2_mid1_radian = np.pi * ((aspect[y, x] - alpha[y, x]/4) / 180)
                    r2_mid2_radian = np.pi * ((aspect[y, x]) / 180)
                    r2_mid3_radian = np.pi * ((aspect[y, x] + alpha[y, x]/4) / 180)
                    r2_end_radian = np.pi * ((aspect[y, x] + alpha[y, x]/2) / 180)
                        
                poly_y_g[0], poly_x_g[0] = get_projected_point(r1[y, x], r1_start_radian, y, x)
                poly_y_g[1], poly_x_g[1] = get_projected_point(r1[y, x], r1_mid1_radian, y, x)
                poly_y_g[2], poly_x_g[2] = get_projected_point(r1[y, x], r1_mid2_radian, y, x)
                poly_y_g[3], poly_x_g[3] = get_projected_point(r1[y, x], r1_mid3_radian, y, x)
                poly_y_g[4], poly_x_g[4] = get_projected_point(r1[y, x], r1_end_radian, y, x)
                poly_y_g[5], poly_x_g[5] = get_projected_point(r2[y, x], r2_start_radian, y, x)
                poly_y_g[6], poly_x_g[6] = get_projected_point(r2[y, x], r2_mid1_radian, y, x)
                poly_y_g[7], poly_x_g[7] = get_projected_point(r2[y, x], r2_mid2_radian, y, x)
                poly_y_g[8], poly_x_g[8] = get_projected_point(r2[y, x], r2_mid3_radian, y, x)
                poly_y_g[9], poly_x_g[9] = get_projected_point(r2[y, x], r2_end_radian, y, x)

                # calculate slope max in polygon
                max_slope[y, x] = rasterize_polygon(poly_y_g, poly_x_g, fold, slope, y, x)

        return max_slope
