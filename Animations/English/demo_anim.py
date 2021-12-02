from manim import *
from textClasses import *
from utils import *

class Demonstration(Scene):
    def construct(self):
        # Chapter intro
        titre = SplashScreen(title="3 - Demonstration of use", titleScale=.5, titleShift=2*UP)
        self.play(FadeIn(titre))
        self.wait(1)

        text1 = TextItem("Thanks to the saving phase, implementing new features is convenient").shift(UP)
        text2 = TextItem("Interactive version of the Spate tool")
        text3 = TextItem("3D visualization to better apprehend the environment").shift(DOWN)

        text = Group(text1, text2, text3).scale(.5).shift(7*LEFT)
        for t in text:
            self.play(FadeIn(t))
            if t == text1:  self.wait(5)
            elif t == text2:    self.wait(3)
            else:   self.wait(5)