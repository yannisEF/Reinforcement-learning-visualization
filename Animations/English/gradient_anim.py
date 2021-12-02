import random
import math

from manim import *
from textClasses import *
from utils import *

class EtudeGradient(Scene):
    def construct(self):
        # Title
        titre = SplashScreen(title="1 - Methods of visualization", titleScale=.5, titleShift=3*UP)
        subtitle = Text("Gradient study").scale(.35).shift(2.5*UP)
        self.add(titre)
        self.play(FadeIn(subtitle))
        self.wait(2)

        # Intro
        text1 = TextItem(text="Follow a model's gradient descent").shift(UP)
        text2 = TextItem(text="Detect the structures it went through")
        text3 = TextItem(text="Compare the directions taken between each step").shift(DOWN)

        text = Group(text1, text2, text3).scale(.5).shift(4*LEFT)
        for t in text:
            self.play(FadeIn(t))
            if t == text1:  self.wait(4)
            self.wait(2)
        
        self.wait(2)
        self.play(FadeOut(text))
        self.wait(2)

        # Gradient descent
        #   Point seul
        group1 = VGroup()
        points = makeRandomPoints(6, center=DOWN, minX=-2, maxX=2, minY=-1, maxY=2)
        point = Dot(points[0])
        self.add_foreground_mobjects(point)

        self.play(FadeIn(point))
        self.wait(1)
        group1.add(point)

        path = makePath(points)
        for line in path:
            history = [Dot(line.start, color=RED), Dot(line.end, color=GREEN)]
            self.play(*[FadeIn(p) for p in history])
            
            self.play(MoveAlongPath(point, line))

            group1.add(*history)

            for p in history:   p.set_opacity(.5)
        self.wait(2)
        self.play(group1.animate.shift(3*LEFT))
        self.wait(4)

        #   Point avec trajectoires
        history[-1].set_color(RED)
        point.set_center(points[0])
        points = [shiftList(p, 3*LEFT) for p in points]
        path = makePath(points)
        
        images = (ImageMobject("Images/gradient5lines/gradientStudy5lines{}.png".format(k)).shift(3.5*RIGHT).scale(.5).shift(.5*DOWN) for k in range(1,6+1))
        image = None
        pImages = []

        traceLines = []
        for line in path:
            history = [Dot(line.start, color=RED), Dot(line.end, color=GREEN)]
            self.play(*[FadeIn(p) for p in history])

            dashedLine = VMobject()
            self.add(dashedLine)

            tempImg = next(images)
            animImg = FadeIn(tempImg) if image is None else Transform(image, tempImg)
            image = tempImg
            pImages.append(image)

            dashedLine.add_updater(lambda x: x.become(DashedLine(history[0], point.get_center(), color=ORANGE)))
            self.play(MoveAlongPath(point, line), animImg)
            self.remove(dashedLine)

            traceLines.append(DashedLine(history[0], point.get_center(), color=ORANGE))
            self.add(traceLines[-1])
            group1.add(line, *history)

            for p in history:   p.set_opacity(.5)
            self.wait(1)
        self.wait(12)

        #   Showing angles on the side
        angles = []
        for k in range(len(traceLines)-1):
            distMin = min(distance(traceLines[k].start, traceLines[k].end), distance(traceLines[k+1].start, traceLines[k+1].end))
            angle = Angle(traceLines[k], traceLines[k+1], quadrant=(-1,1), radius=distMin/5).set_color(GREEN)
            angles.append(angle)

        group1.add(*angles)
        self.play(Transform(image, next(images)), *[Create(a) for a in angles])
        self.wait(5)

        # Conclusion gradient
        exemple = ImageMobject("Images/gradientStudy3.png").scale(.25)
        descr = Text("Swimmer environment, SAC algorithm from 250 to 10,000 steps").scale(.1725/1.5).shift(1.6*DOWN)

        exempleGroup = Group(exemple, descr).shift(3.5*RIGHT).scale(1.5).shift(.5*DOWN)

        self.play(*[FadeOut(i) for i in pImages])
        self.play(FadeIn(exempleGroup))
        self.wait(22)

        self.play(*[FadeOut(o) for o in group1], *[FadeOut(l) for l in traceLines], FadeOut(dashedLine), *[FadeOut(l) for l in path])
        self.play(exempleGroup.animate.shift(7*LEFT))
        self.wait(2)

        conc1 = TextItem("We retrieve the change in angle between each direction").shift(.5*UP)
        conc2 = TextItem("Brings information about the model's gradient descent").shift(.5*DOWN)

        concGroup = Group(conc1, conc2).scale(.5).shift(1.5*LEFT + .5*DOWN)

        for c in concGroup:
            self.play(FadeIn(c))
            self.wait(3)
        self.wait(3)
        self.play(FadeOut(subtitle))