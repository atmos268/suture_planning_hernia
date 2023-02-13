
import scipy_generate_sample_spline
import SuturePlacer

def suture_placing_pipeline(SuturePlacer):
    # TODO Varun: will rope in Sam's code that has the interface for the surgeon to click
    #  points along the wound. That'll return a spline.

    # But for now, just use this sample spline. Its a Bezier spline

    # Varun/Viraj: For now this is OK, but maybe we will need to incorporate wounds that can't be represented as y(x) later using multiple B-spline curves or something else.
    """ Notes on old version of Bezier: So this bezier library can make arbitrary parametric [t -> x(t), y(t)] bezier curves which allows for wounds where y is not a function of x or vice versa,
    #  but I don't think it has a function to fit points to a bezier curve. SciPy's bezier module can fit points to a curve, but it is in the format [x -> y] which is more limiting
    #  for the types of curves we can handle. Goal is to fit points to a parametric bezier curve.
    """
    wound = scipy_generate_sample_spline.generate_sample_spline()

    # Put the wound into all the relevant objects
    SuturePlacer.wound = wound
    SuturePlacer.DistanceCalculator.wound = wound
    SuturePlacer.Optimizer.wound = wound

    # The main algorithm
    SuturePlacer.place_sutures()

if __name__ == "__main__":
    SuturePlacer = SuturePlacer()
    suture_placing_pipeline(SuturePlacer)
