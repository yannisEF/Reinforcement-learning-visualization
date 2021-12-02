from manim import *

from utils import *
from textClasses import *

class Graphique(Scene):
    def construct(self):
        # Title
        titre = SplashScreen(title="2 - Structures of the developped tools", titleScale=.5, titleShift=2*UP)
        self.add(titre)

        subTitle = Text("Saving phase").scale(.35).shift(1.5*UP)
        titre.add(subTitle)
        self.play(FadeIn(subTitle))
        self.wait(1)

        text1 = TextItem("Saves the data for post processing")
        text2 = TextItem("Allows to personalize the output").shift(DOWN)
        text3 = TextItem("Memory heavy, LZMA compression format slow to compress but fast to decompress").shift(2*DOWN)

        text = Group(text1, text2, text3).scale(.5).shift(7*LEFT+UP)

        for t in text:
            self.play(FadeIn(t))
            self.wait(2)
        self.wait(18)

        self.play(FadeOut(text), titre.animate.shift(UP))
        self.wait(1)

        # Images
        scaleGrad, scaleVignette = .3, .5

        grad1 = ImageMobject("Images/gradientStudy2.png").scale(scaleGrad).shift(4.5*LEFT)
        grad2 = ImageMobject("Images/gradientStudy3.png").scale(scaleGrad)
        grad3 = ImageMobject("Images/gradientStudy1.png").scale(scaleGrad).shift(4.5*RIGHT)
        gradDescr = Text("Swimmer environment, SAC algorithm from 250 to 10,000 steps").scale(.2).shift(2*DOWN)
        gradGroup = Group(grad1, grad2, grad3, gradDescr)

        vign1 = ImageMobject("Images/vignette2D2.png").scale(scaleVignette).shift(2*UP)
        vign2 = ImageMobject("Images/vignette2D1.png").scale(scaleVignette)
        vign3 = ImageMobject("Images/vignette2D3.png").scale(scaleVignette).shift(2*DOWN)
        vignDescr = Text("Pendulum environment, SAC algorithm at 5000 steps").scale(.2).shift(3.2*DOWN)
        vignGroup = Group(vign1, vign2, vign3, vignDescr)

        # Gradient
        self.play(FadeIn(grad1), FadeIn(gradDescr))
        self.wait(1)
        self.play(TransformFromCopy(grad1, grad2))
        self.wait(1)
        self.play(TransformFromCopy(grad2, grad3))
        self.wait(25)

        #   Analyse Swimmer
        self.play(FadeOut(gradGroup))
        #self.play(grad2.animate.scale(2), FadeOut(titre), gradDescr.animate.shift(1.65*DOWN))
        self.wait(1)
        
        # Vignette
        self.play(FadeIn(vign1), FadeIn(vignDescr), FadeOut(titre))
        self.wait(1)
        self.play(TransformFromCopy(vign1, vign2))
        self.wait(1)
        self.play(TransformFromCopy(vign2, vign3))
        self.wait(10)

        #   Analyse Pendulum
        self.play(FadeOut(vignGroup))
        self.wait(1)
