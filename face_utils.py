# normalize circles coordinate and radius
def normalize_circle(image, coord, radius, maxRadius, minRadius):
    # point_x / max_x and point_y / max_y for the image points
    # coordinates are of form (x,y) but image is (y,x)
    radius_norm = (radius-minRadius)/(maxRadius-minRadius)
    coord_norm = (coord[0]/image.shape[1], coord[1]/image.shape[0])
    offset = (coord_norm[0] - .5, coord_norm[1] - .5)
    return coord_norm, radius_norm, offset
