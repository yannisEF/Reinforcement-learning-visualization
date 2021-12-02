import random
import math

from manim import *
from textClasses import *
from utils import *

class Structure(Scene):
    def moveAlong(self, point, path, trace=False, color=ORANGE):
        traceLines = []
        for line in path:
            if trace is True:
                dashedLine = VMobject()
                self.add(dashedLine)

                dashedLine.add_updater(lambda x: x.become(DashedLine(line.start, point.get_center(), color=color)))
                self.play(MoveAlongPath(point, line))

                traceLines.append(DashedLine(line.start, line.end, color=color))
                self.add(traceLines[-1])

            else:
                self.play(MoveAlongPath(point, line))

        return traceLines

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
        # Chapter Intro
        titre = SplashScreen(title="2 - Structure des outils développés", titleScale=.5, titleShift=2*UP)

        text1 = TextItem("Outils utilisables de l'entraînement jusqu'à l'affichage")
        text2 = TextItem("Possibilité de rentrer ses propres politiques, à condition d'utiliser les bons formats").shift(DOWN)
        text3 = TextItem("Structure modulable, l'utilisateur peut facilement implémenter ses propres fonctionnalités").shift(2*DOWN)

        text = Group(text1, text2, text3).scale(.5).shift(7*LEFT+UP)

        self.play(FadeIn(titre))
        self.wait(2)

        for t in text:
            self.play(FadeIn(t))
            if t == text1:  self.wait(18)
            elif t == text2:   self.wait(7)
            else:   self.wait(3)
        self.wait(2)

        # SB3 portage
        subtitle = Text("Portage à Stable-baselines-3").scale(.35).shift(1.5*UP)
        self.play(FadeIn(subtitle), FadeOut(text))
        titre.add(subtitle)
        self.wait(2)

        text1 = TextItem("Réécriture du code des années précédentes")
        text2 = TextItem("Portage à Stable-baselines-3, librairie aux diverses implémentations et au code documenté").shift(DOWN)
        text3 = TextItem("L'utilisateur peut choisir l'environnement et l'algorithme utilisé").shift(2*DOWN)
        
        text = Group(text1, text2, text3).scale(.5).shift(6.5*LEFT+UP+.5*DOWN)

        for t in text:
            self.play(FadeIn(t))
            if t == text1:  self.wait(3)
            elif t == text2:    self.wait(5)
            else:   self.wait(3)
        self.wait(2)
        self.play(FadeOut(text))

        subtitle2 = Text("Processus d'utilisation des outils").scale(.35).shift(1.5*UP)
        titre.remove(subtitle)
        self.play(FadeOut(subtitle))
        titre.add(subtitle2)

        self.play(FadeIn(subtitle2))
        self.play(titre.animate.shift(UP))
        self.wait(1)


        # Phases de calcul

        init_scale = .9
        img_scale = .65
        radius_vignette = .85
        boxesOffset = DOWN
        boxScale = .9

        # Phase de préparation
        prepBorders = RoundedRectangle(height = 6)
        prepTitle = Text("Phase de préparation").scale(.25)
        prepTitle.shift(2.65*UP + .8*LEFT)

        prepText = Text("Entraînement du modèle, enregistrement de \nla descente de gradient à une fréquence régulière.").scale(.2)
        prepText.shift(1.5*UP)

        image = ImageMobject("Images/Background_perlin.png").scale(img_scale)
        image.shift(DOWN)

        point = Dot(DOWN, color=RED, radius=img_scale*DEFAULT_DOT_RADIUS)
        self.add_foreground_mobjects(point)

        preparation = Group(prepBorders, prepTitle, prepText, image, point).scale(init_scale)
        preparation.shift(4.5*LEFT)

        # Phase de calcul
        calcBorders = RoundedRectangle(height = 6)
        calcTitle = Text("Phase de calcul").scale(.25)
        calcTitle.shift(2.65*UP + .8*LEFT)

        calcText = Text("Echantillonnage de l'espace, application des méthodes \nd'étude de gradient ou de Vignette.").scale(.2)
        calcText.shift(1.5*UP)

        calcul = Group(calcBorders, calcTitle, calcText).scale(init_scale)

        # Phase de sauvegarde
        savBorders = RoundedRectangle(height = 6)
        savTitle = Text("Phase de sauvegarde").scale(.25)
        savTitle.shift(2.65*UP + .8*LEFT)

        savText = Text("Sauvegarde des données, \nobjets manipulables à postériori.").scale(.2)
        savText.shift(1.5*UP)

        gradBox = RoundedRectangle(width=2, height=1, color=RED).set_fill(RED)
        gradText = Text("SavedGradient").scale(.25)
        gradSav = Group(gradBox, gradText)

        vignBox = RoundedRectangle(width=2, height=1, color=RED).set_fill(RED)
        vignText = Text("SavedVignette").scale(.25)
        vignSav = Group(vignBox, vignText).shift(1.75*DOWN)

        self.add_foreground_mobjects(gradSav, vignSav)
        sauvegarde = Group(savBorders, savTitle, savText, gradSav, vignSav).scale(init_scale)
        sauvegarde.shift(4.5*RIGHT)

        group = Group(preparation, calcul, sauvegarde).shift(boxesOffset).scale(boxScale)
        self.play(FadeIn(group))

        # Cycle present
        self.wait(1)
        self.play(preparation.animate.scale(1/init_scale))
        self.wait(3)
        self.play(preparation.animate.scale(init_scale))

        self.wait(1)
        self.play(calcul.animate.scale(1/init_scale))
        self.wait(2)
        self.play(calcul.animate.scale(init_scale))

        self.wait(1)
        self.play(sauvegarde.animate.scale(1/init_scale))
        self.wait(2)
        self.play(sauvegarde.animate.scale(init_scale))
        self.wait(1)

        # Process preparation
        self.play(preparation.animate.scale(1/init_scale))

        points = [point.get_center()] + makeRandomPoints(9, center=point.get_center(),
                    minX=-2*img_scale, maxX=2*img_scale, minY=-2*img_scale, maxY=2*img_scale)
        path = makePath(points)
        traceLines = self.moveAlong(point, path, trace=True)
        preparation.add(*traceLines)

        self.wait(1)
        self.play(preparation.animate.scale(init_scale))
        
        # Process calcul
        self.wait(1)
        self.play(calcul.animate.scale(1/init_scale))

        traceLines2 = [l.copy() for l in traceLines]
        pathAndPoint = VGroup(point.copy(), *traceLines2).scale(1/init_scale)
        self.play(pathAndPoint.animate.shift((4.5*RIGHT+UP)*boxScale).scale(.75))

        for k in range(len(traceLines2)-1):
            angle = Angle(traceLines2[k], traceLines2[k+1], quadrant=(-1,1), radius=distance(traceLines2[k+1].start, traceLines2[k+1].end)/5).set_color(GREEN)
            #self.play(Create(angle))
            #pathAndPoint.add(angle)

        calcul.add(pathAndPoint)
        self.wait(5)

        centreVignette = Dot(1.75*DOWN*boxScale+boxesOffset, color=BLUE)
        self.add_foreground_mobjects(centreVignette)

        cercle = Circle(radius_vignette*boxScale, arc_center=centreVignette.get_center(), color=RED)
        angles = (40, 100, 165, 205)
        linesVignette = self.makeLines(angles, center=centreVignette.get_center(), radius=radius_vignette*boxScale, wait=None)
        self.play(GrowFromCenter(cercle), *[GrowFromPoint(l, centreVignette.get_center()) for l in linesVignette], FadeIn(centreVignette))

        vignette = Group(cercle, *linesVignette, centreVignette)
        calcul.add(vignette)

        self.wait(8)
        self.play(calcul.animate.scale(init_scale))

        # Process sauvegarde
        self.wait(1)
        self.play(sauvegarde.animate.scale(1/init_scale))
        
        pathAndPointCopy = pathAndPoint.copy()
        vignetteCopy = vignette.copy()
        self.play(pathAndPointCopy.animate.shift(4.5*RIGHT*boxScale).scale(0), vignetteCopy.animate.shift(4.5*RIGHT*boxScale).scale(0))
        self.remove(pathAndPointCopy, vignetteCopy)

        self.wait(8)
        self.play(sauvegarde.animate.scale(init_scale))

        self.wait(2)