
import scipy_generate_sample_spline
import scipy.interpolate as inter
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

    x = [0.0, 0.7, 1.0, 1.1, 1.6, 1.8, 2]
    y = [0.0, 0.5, 1.8, 0.9, 0.4, 0.8, 1.2]
    deg = 3

    # couldn't find reference to this in the codebase? I'm using make_interp_spline for now
    # wound = scipy_generate_sample_spline.generate_sample_spline()
    tck, u = inter.splprep([x, y], k=deg)
    wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

    # Put the wound into all the relevant objects
    SuturePlacer.wound_parametric = wound_parametric
    SuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
    SuturePlacer.Optimizer.wound_parametric = wound_parametric

    # The main algorithm
    SuturePlacer.place_sutures()

if __name__ == "__main__":
    SuturePlacer = SuturePlacer.SuturePlacer()
    suture_placing_pipeline(SuturePlacer)
