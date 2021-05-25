import math
import numpy as np

from Recolocate import Recolocate
from manim import *
from utils import *
from textClasses import *

class MethodeFaisceaux(Scene):   
    def drawPath(self, points, show=True, wait=None, nDash=0, opacity=1, color=ORANGE):
        lines = []
        for k in range(len(points)-1):
            if nDash == 0:
                lines.append(Line(points[k], points[k+1], color=color))
            else:
                d = distance(points[k].get_center(), points[k+1].get_center())
                lines.append(DashedLine(points[k], points[k+1], color=color, dashed_length=.5*d/nDash))

            lines[-1].set_opacity(opacity)
            if show is True:
                if wait is not None:
                    self.play(Create(lines[-1]))
                else:
                    self.add(lines[-1])
        return lines

    def makeLines(self, angles, center=[0,0,0], radius=1, numberDash=10, wait=.5, color=ORANGE):
        Lines = [makeLineAtAngle(a, center=center, radius=radius, numberDash=numberDash, color=color) for a in angles]

        if wait is not None:
            if wait == 0:
                self.play(*[GrowFromPoint(line, center) for line in Lines])
            else:
                for line in Lines:
                    self.play(GrowFromPoint(line, center))
                    self.wait(wait)
        return Lines

    def construct(self):
        # Chapter intro
        titre = SplashScreen(title="4 - Développements futurs", titleScale=.5, titleShift=2*UP)
        self.play(FadeIn(titre))
        self.wait(5)

        # Vitesse des calculs
        subTitle = Text("Vitesse des calculs").scale(.35).shift(1.5*UP)
        titre.add(subTitle)
        self.play(FadeIn(subTitle))
        self.wait(4)

        text1 = TextItem("A cause du problème de généralisation, il est nécessaire de calculer les performances des échantillons un certain nombre de fois").shift(UP)
        text2 = TextItem("Compromis entre la précision souhaitée et la puissance de calcul disponible")
        text3 = TextItem("Solutions envisagées :").shift(DOWN)
        text4 = Group(TextItem("exécution multi-coeurs"),
                    TextItem("possibilité de sauvegarder des points de contrôle (checkpoint) des exécutions").shift(.7*DOWN),
                    TextItem("calculs en plusieurs passes -> observation de résultats de plus en plus précis").shift(2*.7*DOWN))
        text4.shift(2*DOWN+RIGHT)

        text = Group(text1, text2, text3, text4).scale(.5).shift(7*LEFT)
        for t in text:
            self.play(FadeIn(t))
            if t == text1:  self.wait(20)
            elif t == text2:    self.wait(10)
            elif t == text3:    self.wait(1)
            elif t == text4:    self.wait(50)

        self.play(FadeOut(text), titre.animate.shift(UP))
        self.play(FadeOut(subTitle))
        self.wait(2)

        # Méthode des faisceaux
        subTitle2 = Text("Méthode des faisceaux").scale(.35).shift(2.5*UP)
        titre.add(subTitle2)
        self.play(FadeIn(subTitle2))
        self.wait(3)

        # Demonstration
        points = Group(*[Dot(color=BLUE, radius=2*DEFAULT_DOT_RADIUS) for _ in range(7)])
        points[0].shift(5*LEFT + 2*DOWN)
        points[1].shift(4*LEFT + UP)
        points[2].shift(2*LEFT + 2*UP)
        points[3].shift(RIGHT)
        points[4].shift(2*RIGHT + 2*UP)
        points[5].shift(6*RIGHT + UP)
        points[6].shift(5*RIGHT + 2*DOWN)

        points.shift(.5*DOWN).scale(.75)

        path = self.drawPath(points, show=False, nDash=10, color=GREEN, opacity=.5)
        for k in range(len(points)):
            if k > 0:
                self.play(FadeIn(points[k]), Create(path[k-1]))
            else:
                self.play(FadeIn(points[k]))

        lines = self.drawPath(points, wait=1)
        self.wait(5)
        self.play(*[FadeOut(l) for l in lines])

        lines = self.drawPath(points[::2], wait=1)
        self.wait(2)
        self.play(*[FadeOut(l) for l in lines])

        lines = self.drawPath(points[::3], wait=1)
        self.wait(8)
        self.play(*[FadeOut(l) for l in lines])

        removeGroup = VGroup(*points[-3:], *path[-3:])
        self.play(FadeOut(removeGroup))
        self.wait(1)

        points, path = points[:-3], path[:-3]
        stayGroup = VGroup(*points, *path)
        self.play(Recolocate(stayGroup, points[0], points[-1], 3*LEFT+1.5*DOWN, 3*RIGHT+1.5*DOWN))
        self.wait(13)
        d = distance(points[0].get_center(), points[-1].get_center())

        # Vignette
        circle = Circle(d, arc_center=points[0].get_center(), color=RED)
        angles = (0, angleBetween(points[0].get_center(), points[1].get_center()),
                angleBetween(points[0].get_center(), points[2].get_center()))
        lines = self.makeLines(angles, center=points[0].get_center(), radius=d, numberDash=20, wait=None)

        self.play(GrowFromCenter(circle), *[GrowFromPoint(l, points[0].get_center()) for l in lines])
        self.wait(3)

        self.play(ShrinkToCenter(circle), *[l.animate.scale(0.001) for l in lines])
        self.remove(circle, *lines)
        self.wait(10)

        # Faisceaux
        line = Line(points[0].get_center(), points[-1].get_center(), color=ORANGE)
        self.play(Create(line))
        self.wait(1)

        hyperplans = [Line(points[0].get_center()+2*DOWN, points[0].get_center()+2*UP, color=GREEN),
                    Line(points[-1].get_center()+2*DOWN, points[-1].get_center()+2*UP, color=GREEN)]
        self.play(*[Create(h) for h in hyperplans])
        self.wait(4)

        sampled1 = sampleLine(hyperplans[0], 6)
        sampled2 = sampleLine(hyperplans[1], 6)
        self.play(*[FadeIn(s) for s in sampled1], *[FadeIn(s) for s in sampled2])
        self.wait(3)

        beam = [DashedLine(sampled1[k].get_center(), sampled2[k].get_center(),
                            color=ORANGE, dash_length=.5*d/20) for k in range(len(sampled1))]
        self.play(*[Create(b) for b in beam])
        self.wait(3)

        # Cela revient à concentrer Vignette
        drawGroup = Group(stayGroup, line, *hyperplans, *sampled1, *sampled2, *beam)
        self.play(drawGroup.animate.scale(.75))
        self.play(drawGroup.animate.shift(3*LEFT))

        descr1 = TextItem("Cette méthode revient à concentrer Vignette selon une direction").shift(UP)
        descr2 = TextItem("On obtient un meilleur aperçu de l'environnement rencontré\npar le modèle")
        descr3 = TextItem("Du fait de la concentration des droites, moins de discontinuités pour\nla détection de structures").shift(1.1*DOWN)

        descr = Group(descr1, descr2, descr3).scale(.5).shift(3*LEFT+.25*DOWN)

        for d in descr:
            self.play(FadeIn(d))
            self.wait(3)
        self.wait(3)