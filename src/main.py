import coefficient_calculators as coefficient_calculators
import topography_calculators as topography_calculators
import risk_calculators as risk_calculators
from layer_calculator_factory import LayerCalculatorFactory
from layer import Layer
from map import Map, Region

def create_standard_algorthm_factory():
    factory = LayerCalculatorFactory()

    # coefficient layers
    factory.register_calculator(Layer.COEFFICIENT, coefficient_calculators.EvansCoefficientCalculator)
    
    # topo layers
    factory.register_calculator(Layer.SLOPE, topography_calculators.SlopeCalculator)
    factory.register_calculator(Layer.PROFILE_CURVATURE, topography_calculators.ProfileCurvatureCalculator)
    factory.register_calculator(Layer.PLAN_CURVATURE, topography_calculators.PlanCurvatureCalculator)
    factory.register_calculator(Layer.SHAPE, topography_calculators.ShapeCalculator)
    
    # risk layers
    factory.register_calculator(Layer.SITUATION_RISK, risk_calculators.SituationRiskCalculator)
    factory.register_calculator(Layer.TOPO_RISK, risk_calculators.TopoRiskStandardCalculator)
    factory.register_calculator(Layer.TOPO_RISK_LEVEL, risk_calculators.TopoRiskLevelCalculator)
    factory.register_calculator(Layer.COMBINED_RISK, risk_calculators.CombinedRiskCalculator)

    return factory

def create_ATHM_algorithm_factory():
    factory = LayerCalculatorFactory()

    # coefficient layers
    factory.register_calculator(Layer.COEFFICIENT, coefficient_calculators.EvansCoefficientCalculator)
    
    # topo layers
    factory.register_calculator(Layer.SLOPE, topography_calculators.SlopeCalculator)
    factory.register_calculator(Layer.ASPECT, topography_calculators.AspectCalculator)
    factory.register_calculator(Layer.PLAN_CURVATURE, topography_calculators.PlanCurvatureCalculator)
    factory.register_calculator(Layer.FOLD, topography_calculators.FoldCalculator)

    # rsa layers
    factory.register_calculator(Layer.RELEVANT_SLOPE_AREA, topography_calculators.RelevantSlopeAreaCalculator)
    factory.register_calculator(Layer.RELEVANT_SLOPE_AREA_PROPERTIES, topography_calculators.RelevantSlopeAreaPropertiesCalculator)
    
    # risk layers
    factory.register_calculator(Layer.SITUATION_RISK, risk_calculators.SituationRiskCalculator)
    factory.register_calculator(Layer.TOPO_RISK, risk_calculators.TopoRiskAthmCalculator)
    factory.register_calculator(Layer.TOPO_RISK_LEVEL, risk_calculators.TopoRiskLevelCalculator)
    factory.register_calculator(Layer.PROFILE_CURVATURE, risk_calculators.CombinedRiskCalculator)

    return factory

def create_topo_algorithm_factory():
    factory = LayerCalculatorFactory()

    # coefficient layers
    factory.register_calculator(Layer.COEFFICIENT, coefficient_calculators.EvansCoefficientCalculator)
    
    # topo layers
    factory.register_calculator(Layer.SLOPE, topography_calculators.SlopeCalculator)
    factory.register_calculator(Layer.ASPECT, topography_calculators.AspectCalculator)
    factory.register_calculator(Layer.PROFILE_CURVATURE, topography_calculators.ProfileCurvatureCalculator)
    factory.register_calculator(Layer.PLAN_CURVATURE, topography_calculators.PlanCurvatureCalculator)
    factory.register_calculator(Layer.FOLD, topography_calculators.FoldCalculator)

    # topo output layers
    factory.register_calculator(Layer.HEIGHT_LEVEL, topography_calculators.HeightLevelCalculator)
    factory.register_calculator(Layer.SLOPE_LEVEL, topography_calculators.SlopeLevelCalculator)
    factory.register_calculator(Layer.ASPECT_LEVEL, topography_calculators.AspectLevelCalculator)
    factory.register_calculator(Layer.SHAPE, topography_calculators.ShapeCalculator)
    factory.register_calculator(Layer.FOLD_LEVEL, topography_calculators.FoldLevelCalculator)

    return factory

def create_all_factory(standard = True):
    factory = LayerCalculatorFactory()

    # coefficient layers
    factory.register_calculator(Layer.COEFFICIENT, coefficient_calculators.EvansCoefficientCalculator)
    
    # topo layers
    factory.register_calculator(Layer.SLOPE, topography_calculators.SlopeCalculator)
    factory.register_calculator(Layer.ASPECT, topography_calculators.AspectCalculator)
    factory.register_calculator(Layer.PROFILE_CURVATURE, topography_calculators.ProfileCurvatureCalculator)
    factory.register_calculator(Layer.PLAN_CURVATURE, topography_calculators.PlanCurvatureCalculator)
    factory.register_calculator(Layer.FOLD, topography_calculators.FoldCalculator)

    factory.register_calculator(Layer.HEIGHT_LEVEL, topography_calculators.HeightLevelCalculator)
    factory.register_calculator(Layer.SLOPE_LEVEL, topography_calculators.SlopeLevelCalculator)
    factory.register_calculator(Layer.ASPECT_LEVEL, topography_calculators.AspectLevelCalculator)
    factory.register_calculator(Layer.SHAPE, topography_calculators.ShapeCalculator)
    factory.register_calculator(Layer.FOLD_LEVEL, topography_calculators.FoldLevelCalculator)

    factory.register_calculator(Layer.RELEVANT_SLOPE_AREA, topography_calculators.RelevantSlopeAreaCalculator)
    factory.register_calculator(Layer.RELEVANT_SLOPE_AREA_PROPERTIES, topography_calculators.RelevantSlopeAreaPropertiesCalculator)
    
    # risk layers
    factory.register_calculator(Layer.SITUATION_RISK, risk_calculators.SituationRiskCalculator)
    if standard == True:
        factory.register_calculator(Layer.TOPO_RISK, risk_calculators.TopoRiskStandardCalculator)
    else:
        factory.register_calculator(Layer.TOPO_RISK, risk_calculators.TopoRiskAthmCalculator) 
    
    factory.register_calculator(Layer.TOPO_RISK_LEVEL, risk_calculators.TopoRiskLevelCalculator)
    factory.register_calculator(Layer.COMBINED_RISK, risk_calculators.CombinedRiskCalculator)

    return factory

if __name__ == "__main__":

    standard_algorithm_factory = create_standard_algorthm_factory()
    athm_algorithm_factory = create_ATHM_algorithm_factory()
    topo_algorithm_factory = create_topo_algorithm_factory()
    all_factory = create_all_factory(True)

    map = Map(all_factory)
    map.register_new_data()
    map.set_working_region(Region.SWITZERLAND)
    
    map.calculateLayer(Layer.SLOPE)
    map.calculateLayer(Layer.SLOPE_LEVEL)
    map.calculateLayer(Layer.ASPECT)
    map.calculateLayer(Layer.ASPECT_LEVEL)
    map.calculateLayer(Layer.PROFILE_CURVATURE)
    map.calculateLayer(Layer.PLAN_CURVATURE)
    map.calculateLayer(Layer.SHAPE)
    map.calculateLayer(Layer.FOLD)
    map.calculateLayer(Layer.FOLD_LEVEL)
    map.calculateLayer(Layer.HEIGHT_LEVEL)
    map.calculateLayer(Layer.TOPO_RISK)
    map.calculateLayer(Layer.TOPO_RISK_LEVEL)

    map.showLayer([Layer.SLOPE_LEVEL])
    map.showLayer([Layer.ASPECT_LEVEL])
    map.showLayer([Layer.TOPO_RISK])
    map.showLayer([Layer.TOPO_RISK_LEVEL])
    map.showLayer([Layer.RELEVANT_SLOPE_AREA_PROPERTIES.SIZE])
    map.showLayer([Layer.RELEVANT_SLOPE_AREA_PROPERTIES.AVERAGE_SLOPE])
    
    
    # map.show3D([Layer.SHAPE], ['Set1'], [(None, None)], (2100, 5900, 100, 100))