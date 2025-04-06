from aenum import Enum, skip

class Layers(Enum):
    # Input Layers
    HEIGHT = 'height.tif'
    HEIGHT_LEVEL = 'height_level.tif' # OUTPUT

    DANGER_EARLY = 'danger_early.tif'
    DANGER_LATE = 'danger_late.tif'

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
    DIRECTION_LEVEL = 'direction_level.tif' # OUTPUT
    
    SLOPE = 'slope.tif'
    SLOPE_LEVEL = 'slope_level.tif' # OUTPUT
    
    PROFILE_CURVATURE = 'profile_curvature.tif'
    PLAN_CURVATURE = 'plan_curvature.tif'
    SHAPE = 'shape.tif' # OUTPUT

    TOPO_AVALANCHE_RISK = 'topo_avalanche_risk.tif' # OUTPUT
    CURRENT_AVALANCHE_RISK = 'current_avalanche_risk.tif' # OUTPUT
    COMBINED_AVALANCHE_RISK = 'combined_avalanche_risk.tif' # OUTPUT


    