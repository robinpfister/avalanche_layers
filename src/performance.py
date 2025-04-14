import coefficient_calculators as coefficient_calculators
import topography_calculators as topography_calculators
import risk_calculators as risk_calculators
from layer_calculator_factory import LayerCalculatorFactory
from layer import Layer
from map import Map, Region

def create_all_factory():
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
    #factory.register_calculator(Layer.TOPO_RISK, risk_calculators.TopoRiskAthmCalculator) 
    factory.register_calculator(Layer.TOPO_RISK, risk_calculators.TopoRiskStandardCalculator)
    factory.register_calculator(Layer.TOPO_RISK_LEVEL, risk_calculators.TopoRiskLevelCalculator)
    factory.register_calculator(Layer.COMBINED_RISK, risk_calculators.CombinedRiskCalculator)

    return factory

if __name__ == "__main__":

    factory = create_all_factory()

    map = Map(factory)
    map.register_new_data()

    map.set_working_region(Region.BAVARIA)

    map.calculateLayer(Layer.TOPO_RISK_LEVEL)
    

    iterations = 5
    
    # for i in range(iterations):
    #     map.calculateLayer(Layer.SLOPE)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.ASPECT)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.PROFILE_CURVATURE)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.PLAN_CURVATURE)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.FOLD)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.HEIGHT_LEVEL)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.SLOPE_LEVEL)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.ASPECT_LEVEL)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.SHAPE)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.FOLD_LEVEL)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.RELEVANT_SLOPE_AREA)
    # for i in range(iterations):
        # map.calculateLayer(Layer.SITUATION_RISK)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.TOPO_RISK)
    # for i in range(iterations):
    #     map.calculateLayer(Layer.TOPO_RISK_LEVEL)
    # for i in range(iterations):
        # map.calculateLayer(Layer.COMBINED_RISK)
