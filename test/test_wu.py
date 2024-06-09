from numba import njit
import math
import timeit, functools

# # Function to swap two numbers
# def swap(a, b):
#     return b, a

# # Function to return absolute value of number
# def absolute(x):
#     return abs(x)

# # Function to return integer part of a floating point number
# def iPartOfNumber(x):
#     return int(x)

# # Function to round off a number
# def roundNumber(x):
#     return round(x)

# # Function to return fractional part of a number
# def fPartOfNumber(x):
#     return x - iPartOfNumber(x)

# # Function to return 1 - fractional part of number
# def rfPartOfNumber(x):
#     return 1 - fPartOfNumber(x)

# # Function to draw an anti-aliased line
# def getAApixels(x0, y0, x1, y1):
#     xlist, ylist, aalist = [], [], []
#     steep = absolute(y1 - y0) > absolute(x1 - x0)

#     if steep:
#         x0, y0 = swap(x0, y0)
#         x1, y1 = swap(x1, y1)

#     if x0 > x1:
#         x0, x1 = swap(x0, x1)
#         y0, y1 = swap(y0, y1)

#     dx = x1 - x0
#     dy = y1 - y0
#     gradient = dy / dx if dx != 0 else 1

#     xpxl1 = x0
#     xpxl2 = x1
#     intersectY = y0

#     if steep:
#         for x in range(xpxl1, xpxl2 + 1):
#             xlist.append(iPartOfNumber(intersectY))
#             ylist.append(x)
#             aalist.append(rfPartOfNumber(intersectY))
#             # drawPixel(screen, iPartOfNumber(intersectY), x, rfPartOfNumber(intersectY))
#             xlist.append(iPartOfNumber(intersectY)-1)
#             ylist.append(x)
#             aalist.append(fPartOfNumber(intersectY))
#             # drawPixel(screen, iPartOfNumber(intersectY) - 1, x, fPartOfNumber(intersectY))
#             intersectY += gradient
#     else:
#         for x in range(xpxl1, xpxl2 + 1):
#             xlist.append(x)
#             ylist.append(iPartOfNumber(intersectY))
#             aalist.append(rfPartOfNumber(intersectY))
#             # drawPixel(screen, x, iPartOfNumber(intersectY), rfPartOfNumber(intersectY))
#             xlist.append(x)
#             ylist.append(iPartOfNumber(intersectY) - 1)
#             aalist.append(fPartOfNumber(intersectY))
#             # drawPixel(screen, x, iPartOfNumber(intersectY) - 1, fPartOfNumber(intersectY))
#             intersectY += gradient
#     return xlist, ylist, aalist
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
    x = min((thickness/2) - np.abs(x) + 1, 1)
    if x > 1.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    return x

@njit
def wu_line(x0, y0, x1, y1, thickness = 2):
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


# Example usage:


# Example usage
import matplotlib.pyplot as plt
import numpy as np
x0, y0 = 10.25, 10.2
x1, y1 = 80.4, 20.3
x,y,aa = wu_line(x0, y0, x1, y1)
t = timeit.Timer(functools.partial(wu_line, 101.1, 0.5, 0.5, 120.5))
print(t.timeit(100_000))
# x,y,aa = getAApixels(x0, y0, x1, y1)
# coords = np.array([x,y])
pic = np.full((100,100), np.nan)
pic[x, y] = aa
im = plt.imshow(pic)
plt.colorbar(im)
plt.plot([y0, y1], [x0, x1], "-r")
plt.show()
