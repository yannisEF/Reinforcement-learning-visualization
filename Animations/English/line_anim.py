import math

from manim import *
from utils import *
from textClasses import *

class ExplicationLigne(Scene):
    def construct(self):
        # Chapter Intro
        titre = SplashScreen(title="1 - Methods of visualization", titleScale=.5, titleShift=2*UP)

        text1 = TextItem("We want to visualize a glimpse of a N-th dimensional space in 2 or 3 dimensions.")
        text2 = TextItem("Two methods of visualization :")
        text2.add(TextItem("gradient study: observe a trajectory").shift(DOWN+2*RIGHT), TextItem("the Spate method: observe the surroundings").shift(2*DOWN+2*RIGHT))
        text2.shift(DOWN)
        text3 = TextItem("They rely on sampled lines, arranged vertically to create an image.").shift(4*DOWN)

        text = Group(text1, text2, text3).scale(.5).shift(7*LEFT+UP)

        self.wait(1)
        self.play(FadeIn(titre))
        self.wait(6)
        self.play(FadeIn(text1))
        self.wait(10)
        self.play(FadeIn(text2.text), FadeIn(text2.dot))
        self.wait(3)
        self.play(FadeIn(text2[2]))
        self.wait(3)
        self.play(FadeIn(text2[3]))
        self.wait(4)
        self.play(FadeIn(text3))
        self.wait(5)
        self.play(FadeOut(text), titre.animate.shift(UP))

        # Line explaining
        subtitle = Text("An output line of the tools").scale(.35).shift(2.5*UP)
        self.play(FadeIn(subtitle))
        self.wait(1)

        point1 = Dot(2*LEFT, color=BLUE)
        point2 = Dot(2*RIGHT, color=BLUE)

        self.play(GrowFromCenter(point1), GrowFromCenter(point2))
        self.wait(2)

        ligne = Line(point1, point2, color=ORANGE)
        self.play(GrowFromCenter(ligne))
        
        #   Fréquences
        self.wait(1)
        descr1 = Text("Sampling frequency entered by the user").scale(.20).shift(.25*DOWN)
        self.play(FadeIn(descr1))

        d = distance(point1.get_center(), point2.get_center())
        numberDash = (10, 30, 50, 5, 20)
        for n in numberDash:
            self.play(Transform(ligne, DashedLine(point1, point2, color=ORANGE, dash_length=.5*d/n)))
            if n == 10: self.wait(3)
        self.wait(1)
        
        group1 = VGroup(point1, point2, ligne, descr1)
        self.play(group1.animate.shift(.5*UP))

        image = ImageMobject("Images/Ligne.png")
        descr2 = Text("Each sample achieves a certain reward").scale(.20).shift(.25*DOWN)
        group2 = Group(image, descr2).shift(1.5*DOWN)

        self.play(FadeIn(group2))
        self.wait(6)
        self.play(FadeOut(group2), FadeOut(group1), FadeOut(subtitle))
        