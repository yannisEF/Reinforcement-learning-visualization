from manim import *
from textClasses import *
from utils import *

class Demonstration(Scene):
    def construct(self):
        # Chapter intro
        titre = SplashScreen(title="3 - Démonstration de fonctionnement", titleScale=.5, titleShift=2*UP)
        self.play(FadeIn(titre))
        self.wait(1)

        text1 = TextItem("Grâce à la phase de sauvegarde, il est facile d'implémenter de nouvelles fonctionnalités").shift(UP)
        text2 = TextItem("Version interactive de l'outil Vignette")
        text3 = TextItem("Visualisation en 3D pour un meilleure appréhension de l'environnement").shift(DOWN)

        text = Group(text1, text2, text3).scale(.5).shift(7*LEFT)
        for t in text:
            self.play(FadeIn(t))
            if t == text1:  self.wait(5)
            elif t == text2:    self.wait(3)
            else:   self.wait(5)