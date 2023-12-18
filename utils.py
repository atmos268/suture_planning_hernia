# utils
# mallika's code for point clicking
import matplotlib.pyplot as plt

def click_points_simple(img):
    fig = plt.figure()
    plt.imshow(img)
    left_coords,right_coords = [], []
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        nonlocal left_coords,right_coords
        if(event.button==1):
            left_coords.append(coords)
        elif(event.button==3):
            right_coords.append(coords)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords,right_coords