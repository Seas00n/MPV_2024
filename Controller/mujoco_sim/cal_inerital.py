import numpy as np
def getBoxInertia(x, y, z, m, s=None):
    if s is None:
        s = [1, 1, 1]
    x *= s[0]
    y *= s[1]
    z *= s[2]
    xx = 1./12 * m * (y**2 + z**2)
    yy = 1./12 * m * (x**2 + z**2)
    zz = 1./12 * m * (x**2 + y**2)
    return xx, yy, zz

def getSphereInertia(r, m):
    i = 2./5 * m * r**2
    return i, i, i

def getCylinderInertia(r, h, m):
    xx = yy = 1./12 * m * (3 * r**2 + h**2)
    zz = 1./2 * m * r**2
    return xx, yy, zz

if __name__ == '__main__':
    xx, yy, zz = getBoxInertia(0.5, 0.5, 0.5, m=2000)
    print(xx, yy, zz)
    xx, yy, zz = getCylinderInertia(r=0.02, h=0.22, m=1)
    print(xx, yy, zz)
    i,i,i = getSphereInertia(r=0.02, m=0.2)
    print(i,i,i)
