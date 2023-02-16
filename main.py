
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

    x = [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0]
    y = [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3]
    deg = 3

    # couldn't find reference to this in the codebase? I'm using make_interp_spline for now
    # wound = scipy_generate_sample_spline.generate_sample_spline()

    wound = inter.make_interp_spline(x=x, y=y, k=deg, bc_type="clamped" if deg == 3 else None)

    # Put the wound into all the relevant objects
    SuturePlacer.wound = wound
    SuturePlacer.DistanceCalculator.wound = wound
    SuturePlacer.Optimizer.wound = wound

    # The main algorithm
    SuturePlacer.place_sutures()

if __name__ == "__main__":
    SuturePlacer = SuturePlacer.SuturePlacer()
    suture_placing_pipeline(SuturePlacer)
