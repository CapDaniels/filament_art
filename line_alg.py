from numba import njit
import math

@njit
def ipart(x):
    return int(x)
@njit
def round(x):
    return ipart(x + 0.5)
@njit
def fpart(x):
    return x - ipart(x)
@njit
def rfpart(x):
    return 1 - fpart(x)

@njit
def profile(x, thickness):
    # return x
    # return np.clip(min((thickness/2) - np.abs(x) + 1, 1), 0.0, 1.0)
    x = min((thickness/2) - abs(x) + 1, 1)
    if x > 1.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    return x

@njit
def wu_line(x0:float, y0:float, x1:float, y1:float, thickness:float):
    """Wu line algorithm with thickness of the line as an additional input

    Args:
        x0, y0 (float): line start
        x1, y1 (float): line end
        thickness (float): thickness of the line (0 recovers the default Wu line algorithm)

    Returns:
        (list[float], list[float], list[float]): list of x-pixels, y-pixles in the line and value of the pixels 
    """
    xlist, ylist, vallist = [], [], []
    steep = abs(y1 - y0) > abs(x1 - x0)
    
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    dx = x1 - x0
    dy = y1 - y0
    
    gradient = dy / dx if dx != 0 else 1
    ihalf_thickness = math.ceil(thickness) // 2
    
    # Handle first endpoint
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = ipart(yend)
    
    if steep:
        for y in range(-ihalf_thickness, ihalf_thickness + 2):
            if y <= 0:
                # doing this explicitly and not with a function to enable numba support
                xlist.append(ypxl1 + y)
                ylist.append(xpxl1)
                vallist.append(profile(-y + fpart(yend), thickness) * xgap)
            else:
                xlist.append(ypxl1 + y)
                ylist.append(xpxl1)
                vallist.append(profile(y - fpart(yend), thickness) * xgap)
    else:
        for y in range(-ihalf_thickness, ihalf_thickness + 2):
            if y <= 0:
                xlist.append(xpxl1)
                ylist.append(ypxl1 + y)
                vallist.append(profile(-y + fpart(yend), thickness) * xgap)
            else:
                xlist.append(xpxl1)
                ylist.append(ypxl1 + y)
                vallist.append(profile(y - fpart(yend), thickness) * xgap)
    
    intery = yend + gradient
    
    # Handle second endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = fpart(x1 + 0.5)
    xpxl2 = xend
    ypxl2 = ipart(yend)
    
    if steep:
        for y in range(-ihalf_thickness, ihalf_thickness + 2):
            if y <= 0:
                xlist.append(ypxl2 + y)
                ylist.append(xpxl2)
                vallist.append(profile(-y + fpart(yend), thickness) * xgap)
            else:
                xlist.append(ypxl2 + y)
                ylist.append(xpxl2)
                vallist.append(profile(y - fpart(yend), thickness) * xgap)
    else:
        for y in range(-ihalf_thickness, ihalf_thickness + 2):
            if y <= 0:
                xlist.append(xpxl2)
                ylist.append(ypxl2 + y)
                vallist.append(profile(-y + fpart(yend), thickness) * xgap)
            else:
                xlist.append(xpxl2)
                ylist.append(ypxl2 + y)
                vallist.append(profile(y - fpart(yend), thickness) * xgap)
    
    # Main loop
    if steep:
        for x in range(xpxl1 + 1, xpxl2):
            dy = fpart(intery)
            for y in range(-ihalf_thickness, ihalf_thickness + 2):
                if y <= 0:
                    xlist.append(ipart(intery) + y)
                    ylist.append(x)
                    vallist.append(profile(-y + dy, thickness))
                else:
                    xlist.append(ipart(intery) + y)
                    ylist.append(x)
                    vallist.append(profile(y - dy, thickness))
            intery += gradient
    else:
        for x in range(xpxl1 + 1, xpxl2):
            dy = fpart(intery)
            for y in range(-ihalf_thickness, ihalf_thickness + 2):
                if y <= 0:
                    xlist.append(x)
                    ylist.append(ipart(intery) + y)
                    vallist.append(profile(-y + dy, thickness))
                else:
                    xlist.append(x)
                    ylist.append(ipart(intery) + y)
                    vallist.append(profile(y - dy, thickness))
            intery += gradient

    return xlist, ylist, vallist