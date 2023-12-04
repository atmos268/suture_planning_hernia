import numpy as np

def np_arr_to_xyz(pts):

    # make a new file

    # write to it in the right way, notably, points separated by newlines, and coords 
    # separated by whitespace
    with open("xyz_pts.txt", "w") as f:
        for point in pts:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    # handoff to c++


pts_path = 'surrounding_pts.npy'
pts = np.load(pts_path)

np_arr_to_xyz(pts)