from aenum import Enum, skip

class Layer(Enum):
    HEIGHT = 'preprocess_layer/height.tif'
    HEIGHT_LEVEL = 'height_level.tif'

    DANGER_EARLY = 'preprocess_layer/danger_early.tif'
    DANGER_LATE = 'preprocess_layer/danger_late.tif'

    @skip
    class COEFFICIENT(Enum):
        A = 'preprocess_layer/coefficients/a.tif'
        B = 'preprocess_layer/coefficients/b.tif'
        C = 'preprocess_layer/coefficients/c.tif'
        D = 'preprocess_layer/coefficients/d.tif'
        E = 'preprocess_layer/coefficients/e.tif'
        F = 'preprocess_layer/coefficients/f.tif'
        G = 'preprocess_layer/coefficients/g.tif'
        H = 'preprocess_layer/coefficients/h.tif'
        I = 'preprocess_layer/coefficients/i.tif'
        VARIABILITÄT = 'preprocess_layer/variability.tif'
    
    ASPECT = 'preprocess_layer/aspect.tif'
    ASPECT_LEVEL = 'aspect_level.tif'
    
    SLOPE = 'preprocess_layer/slope.tif'
    SLOPE_LEVEL = 'slope_level.tif'
    
    PROFILE_CURVATURE = 'preprocess_layer/profile_curvature.tif'
    PLAN_CURVATURE = 'preprocess_layer/plan_curvature.tif'
    SHAPE = 'shape.tif'

    FOLD = 'preprocess_layer/fold.tif'
    FOLD_LEVEL = 'fold_level.tif'

    @skip
    class RELEVANT_SLOPE_AREA(Enum):
        ALPHA = 'preprocess_layer/rsa/alpha.tif'
        R1 = 'preprocess_layer/rsa/r1.tif'
        R2 = 'preprocess_layer/rsa/r2.tif'
        FORM = 'preprocess_layer/rsa/form.tif'

    class RELEVANT_SLOPE_AREA_PROPERTIES(Enum):
        AVERAGE_SLOPE = 'preprocess_layer/rsa/maximum_slope.tif'
        SIZE = 'preprocess_layer/rsa/size.tif'

    TOPO_RISK = 'topo_risk.tif'
    TOPO_RISK_LEVEL = 'topo_risk_level.tif'
    SITUATION_RISK = 'situation_risk.tif'
    COMBINED_RISK = 'combined_risk.tif'
