import random
import math
import numpy as np

from manim import *

def makePath(points):
    return [Line(points[k], points[k+1]) for k in range(len(points)-1)]

def distance(A,B):
    return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

def makeRandomPoints(number, center=[0,0,0], minX=-4, maxX=4, minRX=None, minY=-2, maxY=2, minRY=None):
    points=[[0,0,0]]
    for _ in range(number):
        x, y = 0,0
        while [x,y,0] in points:
            if minRX is not None:
                while abs(x) < minRX:
                    x = random.uniform(minX,maxX)
            else:
                x = random.uniform(minX,maxX)

            if minRY is not None:
                while abs(y) <= minRY:
                    y = random.uniform(minY,maxY)
            else:
                y = random.uniform(minY,maxY)

        points.append([x + center[0], y + center[1], center[2]])
    return points[1:]

def moveAlong(scene, point, path, trace=False, color=ORANGE):
    traceLines = []
    for line in path:
        if trace is True:
            dashedLine = VMobject()
            scene.add(dashedLine)

            dashedLine.add_updater(lambda x: x.become(DashedLine(line.start, point.get_center(), color=color)))
            scene.play(MoveAlongPath(point, line))

            traceLines.append(DashedLine(line.start, line.end, color=color))
            scene.add(traceLines[-1])
            scene.remove(dashedLine)

        else:
            scene.play(MoveAlongPath(point, line))

    return traceLines

def shiftList(A, shift):
    return [A[k] + shift[k] for k in range(len(shift))]

def distanceToLine(A,line):
    dStart = (A[0]-line.start[0])**2 + (A[1]-line.start[1])**2
    dEnd = (A[0]-line.end[0])**2 + (A[1]-line.end[1])**2

    if dStart <= dEnd:
        return (dStart, "start")
    return (dEnd, "end")

def sampleLine(line, n, color=PURPLE, radius=DEFAULT_DOT_RADIUS):
    points = []
    weights = np.arange(0,1+.5/n,1/n)

    for w in weights:
        if w == 0.5: continue

        p = Dot(line.start * w + line.end * (1-w), color=color, radius=radius)
        points.append(p)
    return points

def makeLineAtAngle(angle, center=[0,0,0], radius=1, numberDash=10, color=ORANGE):
    angle = math.pi * (angle % 360)/180
    p1 = [radius * math.cos(angle) + center[0], radius * math.sin(angle) + center[1], 0 + center[2]]
    p2 = [radius * math.cos(angle+math.pi) + center[0], radius * math.sin(angle+math.pi) + center[1], 0 + center[2]]

    if numberDash is not None:
        return DashedLine(p1,p2, color=color, dash_length=0.5*radius/numberDash)
    return Line(p1,p2)

def angleBetween(A, B):
    adj = A[0] - B[0]
    opp = A[1] - B[1]
    return np.arctan(opp/adj) * 180 / math.pi