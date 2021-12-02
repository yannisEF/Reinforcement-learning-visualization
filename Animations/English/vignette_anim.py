import math
import random

import numpy as np

from manim import *
from utils import *
from textClasses import *


class VignettePortee(Scene):    
    def makeLines(self, angles, center=[0,0,0], radius=1, numberDash=10, wait=.5, color=ORANGE):
        Lines = [makeLineAtAngle(a, center=center, radius=radius, numberDash=numberDash, color=color) for a in angles]

        if wait is not None:
            if wait == 0:
                self.play(*[GrowFromPoint(line, center) for line in Lines])
            else:
                for line in Lines:
                    self.play(GrowFromPoint(line,center))
                    self.wait(wait)
        return Lines

    def construct(self):
        # Title
        titre = SplashScreen(title="1 - Methods of visualization", titleScale=.5, titleShift=3*UP)
        subtitle = Text("The Spate method").scale(.35).shift(2.5*UP)
        self.add(titre)
        self.wait(4)
        self.play(FadeIn(subtitle))
        self.wait(6)

        # Intro
        text1 = TextItem(text="Retrieve a glimpse of the model's surroundings").shift(UP)
        text2 = TextItem(text="Detect the structures around the model's parameters")
        text3 = TextItem(text="Analyse the relative position of other points in the considered environment").shift(DOWN)

        text = Group(text1, text2, text3).scale(.5).shift(6*LEFT)
        for t in text:
            self.play(FadeIn(t))
            if t == text1:  self.wait(2)
            elif t == text2:    self.wait(2)
            else:   self.wait(4)
        
        self.play(FadeOut(text))
        self.wait(1)

        # Vignette
        offset = .5 * DOWN
        p1 = offset
        points = makeRandomPoints(3, center=offset, minRX=.5, minRY=.5, maxX=2, minX=-2)

        maxRadius = math.sqrt(max([(p[0]-offset[0])**2 + (p[1]-offset[1])**2 for p in points]))

        politique = Dot(p1)
        politique.set_fill(BLUE)

        self.play(FadeIn(politique))
        self.wait(5)

        Entree = [Dot(p, color=GREEN) for p in points]
        self.play(*[FadeIn(entree) for entree in Entree])
        self.wait(5)
        
        radius1=1.25
        circle = Circle(radius=radius1).shift(offset)
        self.play(GrowFromCenter(circle))
        self.wait(3)

        angles = [random.randint(0,359) for _ in range(len(points) + 1)]
        Lines = self.makeLines(angles, center=politique.get_center(), wait=0)
        
        self.wait(6)
        Lines2 = self.makeLines(angles, center=politique.get_center(), radius=maxRadius, wait=None)
        self.play(Transform(circle, Circle(radius=maxRadius).shift(offset)), *[Transform(Lines[k], Lines2[k]) for k in range(len(Lines))])

        self.wait(18)
        LinesCopy = Lines[:]
        listTransform = []
        for p in points:
            distanceToLines = [distanceToLine(p, line) for line in LinesCopy]
            minDistance = min(distanceToLines, key=lambda x:x[0])
            closestLine = LinesCopy.pop(distanceToLines.index(minDistance))

            angle = np.arctan((p[1] - offset[1])/(p[0] - offset[0])) * 180 / math.pi
            if minDistance[1] == "end": angle += 180
            listTransform.append(Transform(closestLine, makeLineAtAngle(angle, center=politique.get_center(), radius=maxRadius)))

        self.play(*listTransform)
        self.wait(3)
        
        vignetteGroup = VGroup(politique, *Entree, circle, *Lines)

        self.play(vignetteGroup.animate.scale(.75))
        self.play(vignetteGroup.animate.shift(3*LEFT))

        image2D, image3D = ImageMobject("Images/Vignette_pendulum.png"), ImageMobject("Images/vignette1.png")
        image2D.scale(.75)
        image3D.scale(.75)

        image2D.shift(3*RIGHT+offset)
        image3D.shift(3*RIGHT+offset)
        descr = Text("Example output of the Spate tool, Pendulum environment, SAC algorithm on 5000 steps").scale(.1725).shift(3*RIGHT+offset+1.5*DOWN)

        imgGroup = Group(image2D, descr)
        self.play(FadeIn(imgGroup))
        self.wait(15)
        self.play(Transform(image2D,image3D))
        self.wait(5)

        # Conclusion vignette
        self.play(FadeOut(vignetteGroup), imgGroup.animate.shift(6*LEFT))
        self.wait(2)

        conc1 = TextItem("N-th dimensional learning space").shift(UP)
        conc2 = TextItem("Gives a partial visualization")
        conc3 = TextItem("Get a glimpse of the surrounding structures").shift(DOWN)

        conc = Group(conc1, conc2, conc3).scale(.5).shift(1.5*LEFT + .5*DOWN)
        
        for c in conc:
            self.play(FadeIn(c))
            if c == conc1:  self.wait(4)
            self.wait(1)
        self.wait(5)