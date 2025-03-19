from aenum import Enum, skip

class Layers(Enum):
    HEIGHT = 'height.tif'

    @skip
    class PREPROCESSED_DATA(Enum):
        A = 'preprocessed_data/a_coef.tif'
        B = 'preprocessed_data/b_coef.tif'
        C = 'preprocessed_data/c_coef.tif'
        D = 'preprocessed_data/d_coef.tif'
        E = 'preprocessed_data/e_coef.tif'
        F = 'preprocessed_data/f_coef.tif'
        G = 'preprocessed_data/g_coef.tif'
        H = 'preprocessed_data/h_coef.tif'
        I = 'preprocessed_data/i_coef.tif'
        VARIABILITÄT = 'preprocessed_data/variability.tif'
    
    DIRECTION = 'direction.tif'
    DIRECTION_LEVEL = 'direction_level.tif'
    
    SLOPE = 'slope.tif'
    SLOPE_LEVEL = 'slope_level.tif'
    
    PROFILE_CURVATURE = 'profile_curvature.tif'
    PLAN_CURVATURE = 'plan_curvature.tif'
    SHAPE = 'shape.tif'

    AVALANCHE_RISK = 'avalanche_risk.tif'

    AVALANCHE_REPORT_MASK = 'avalanche_report/avalanche_report_mask.tif'

    @skip
    class AVALANCHE_PROBLEM(Enum):
        WIND_SLAB = 'avalanche_problem/wind_slab.tif'
        NEW_SNOW = 'avalanche_problem/new_snow.tif'
        PERSISTENT_WEAK_LAYER = 'avalanche_problem/persistent_weak_layer.tif'
        NO_DISTINCT_PROBLEM = 'avalanche_problem/no_distinct_problem.tif'

    