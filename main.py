import src.zevenbergen_calculators as zevenbergen_calculators
from src.layer_calculator_factory import LayerCalculatorFactory
from src.layers import Layers
from src.map import Map, Region


if __name__ == "__main__":

    zevenbergen_calculator_factory = LayerCalculatorFactory()
    zevenbergen_calculator_factory.register_calculator(Layers.PREPROCESSED_DATA, zevenbergen_calculators.CalculatorPreprocessing)
    zevenbergen_calculator_factory.register_calculator(Layers.DIRECTION, zevenbergen_calculators.CalculatorDirection)
    zevenbergen_calculator_factory.register_calculator(Layers.SLOPE, zevenbergen_calculators.CalculatorSlope)
    zevenbergen_calculator_factory.register_calculator(Layers.SHAPE, zevenbergen_calculators.CalculatorShape)
    zevenbergen_calculator_factory.register_calculator(Layers.DIRECTION_LEVEL, zevenbergen_calculators.CalculatorDirectionLevel)
    zevenbergen_calculator_factory.register_calculator(Layers.SLOPE_LEVEL, zevenbergen_calculators.CalculatorSlopeLevel)
    zevenbergen_calculator_factory.register_calculator(Layers.PROFILE_CURVATURE, zevenbergen_calculators.CalculatorProfileCurvature)
    zevenbergen_calculator_factory.register_calculator(Layers.PLAN_CURVATURE, zevenbergen_calculators.CalculatorPlanCurvature)
    zevenbergen_calculator_factory.register_calculator(Layers.SHAPE, zevenbergen_calculators.CalculatorShape)
    zevenbergen_calculator_factory.register_calculator(Layers.AVALANCHE_RISK, zevenbergen_calculators.CalculatorAvalancheRisk)

    map = Map(zevenbergen_calculator_factory)
    map.register_new()
    map.set_working_region(Region.BAVARIA)
    map.pull_new()

    # map.calculateLayer(Layers.DIRECTION_LEVEL)

    # map.createAvalancheReportLayersTYRL()

    # map.showLayer([Layers.HEIGHT, Layers.DIRECTION_LEVEL])
    # map.show3D([Layers.DIRECTION_LEVEL], ['viridis', 'viridis', 'Set1', 'viridis'], [(None, None)], (3050, 2200, 100, 100))