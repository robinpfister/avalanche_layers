import coefficient_calculators as coefficient_calculators
import topography_calculators as topography_calculators
import risk_calculators as risk_calculators
from layer_calculator_factory import LayerCalculatorFactory
from layer import Layer
from map import Map, Region, Daytime

def create_algorithm_factory():
    factory = LayerCalculatorFactory()

    # coefficient layers
    factory.register_calculator(Layer.COEFFICIENT, coefficient_calculators.EvansCoefficientCalculator)
    # rsa layers
    factory.register_calculator(Layer.RELEVANT_SLOPE_AREA, topography_calculators.RelevantSlopeAreaCalculator)
    
    # topo basis layers
    factory.register_calculator(Layer.SLOPE, topography_calculators.SlopeCalculator)
    factory.register_calculator(Layer.ASPECT, topography_calculators.AspectCalculator)
    factory.register_calculator(Layer.PROFILE_CURVATURE, topography_calculators.ProfileCurvatureCalculator)
    factory.register_calculator(Layer.PLAN_CURVATURE, topography_calculators.PlanCurvatureCalculator)
    factory.register_calculator(Layer.FOLD, topography_calculators.FoldCloseCalculator)
    factory.register_calculator(Layer.RSA_MAX_SLOPE, topography_calculators.RsaMaxSlopeCalculator)
    # topo level layers
    factory.register_calculator(Layer.HEIGHT_LEVEL, topography_calculators.HeightLevelCalculator)
    factory.register_calculator(Layer.SLOPE_LEVEL, topography_calculators.SlopeLevelCalculator)
    factory.register_calculator(Layer.ASPECT_LEVEL, topography_calculators.AspectLevelCalculator)
    factory.register_calculator(Layer.SHAPE, topography_calculators.ShapeCalculator)

    # risk basis layers
    factory.register_calculator(Layer.TOPO_RISK, risk_calculators.TopoRiskCalculator)
    factory.register_calculator(Layer.COMBINED_RISK, risk_calculators.CombinedRiskCalculator)
    # risk level layers
    factory.register_calculator(Layer.TOPO_RISK_LEVEL, risk_calculators.TopoRiskLevelCalculator)
    factory.register_calculator(Layer.COMBINED_RISK_LEVEL, risk_calculators.CombinedRiskLevelCalculator)

    return factory

if __name__ == "__main__":

    algorithm_factory = create_algorithm_factory()

    map = Map(algorithm_factory)
    map.set_working_daytime(Daytime.EARLY)
    map.register_new_data()
    map.set_working_region(Region.SWITZERLAND)
    map.calculateLayer(Layer.SHAPE)
    map.showLayer([Layer.SHAPE, Layer.SITUATION_RISK])
    map.set_working_region(Region.TYROL)
    map.calculateLayer(Layer.SHAPE)
    map.showLayer([Layer.SHAPE, Layer.SITUATION_RISK])
    map.set_working_region(Region.BAVARIA)
    map.calculateLayer(Layer.SHAPE)
    map.showLayer([Layer.SHAPE, Layer.SITUATION_RISK])