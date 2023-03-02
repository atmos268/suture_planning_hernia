import scipy_generate_sample_spline
import scipy.interpolate as inter
import SuturePlacer
from InsertionPointGenerator import InsertionPointGenerator
import numpy as np
import cv2

def suture_placing_pipeline(SuturePlacer):
    # TODO Varun: will rope in Sam's code that has the interface for the surgeon to click
    #  points along the wound. That'll return a spline.
    space_between_sutures = 0.010  # 1 cm
    desired_compute_time = 1
    IPG = InsertionPointGenerator(cut_width=.0075, desired_compute_time=desired_compute_time,
                                  space_between_sutures=space_between_sutures)

    img_color = cv2.imread('hand_image.png')
    img_point = np.load("record/img_point_inclined.npy")
    sample_spline = False
    if not sample_spline:
        pnts = IPG.get_insertion_points_from_selection(img_color, img_point)
    else:
        pnts = [[46, 233], [50, 213], [57, 195], [67, 175], [77, 160], [91, 136], [107, 114], [121, 111], [137, 111],
         [144, 120], [158, 136], [166, 166], [175, 208], [193, 233], [227, 218], [251, 183], [275, 128]]

    print('pnts\n', pnts)

    # But for now, just use this sample spline. It's a Bezier spline

    # Varun/Viraj: For now this is OK, but maybe we will need to incorporate wounds that can't be represented as y(x) later using multiple B-spline curves or something else.
    """ Notes on old version of Bezier: So this bezier library can make arbitrary parametric [t -> x(t), y(t)] bezier curves which allows for wounds where y is not a function of x or vice versa,
    #  but I don't think it has a function to fit points to a bezier curve. SciPy's bezier module can fit points to a curve, but it is in the format [x -> y] which is more limiting
    #  for the types of curves we can handle. Goal is to fit points to a parametric bezier curve.
    """
    x = [a[0] for a in pnts]
    y = [a[1] for a in pnts]

    # x = [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0] # OLD manually-chosen example
    # y = [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3] # OLD manually-chosen example
    deg = 3

    # couldn't find reference to this in the codebase? I'm using make_interp_spline for now
    # wound = scipy_generate_sample_spline.generate_sample_spline()

    # wound = inter.make_interp_spline(x=x, y=y, k=deg, bc_type="clamped" if deg == 3 else None)

    tck, u = inter.splprep([x, y], s=0)

    def wound(x):
        pnts = inter.splev(x, tck)
        # pnts[1] = pnts[1] * -1
        return pnts

    wound(3)
    # Put the wound into all the relevant objects
    SuturePlacer.wound = wound
    SuturePlacer.tck = tck
    SuturePlacer.DistanceCalculator.wound = wound
    SuturePlacer.DistanceCalculator.tck = tck
    SuturePlacer.Optimizer.wound = wound
    SuturePlacer.Optimizer.tck = tck

    # The main algorithm
    SuturePlacer.place_sutures()

if __name__ == "__main__":
    SuturePlacer = SuturePlacer.SuturePlacer()
    suture_placing_pipeline(SuturePlacer)
    cv2.destroyAllWindows()
