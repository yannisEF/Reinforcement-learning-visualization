from manim import *

from utils import *
from textClasses import *

class Graphique(Scene):
    def construct(self):
        # Title
        titre = SplashScreen(title="2 - Structure des outils développés", titleScale=.5, titleShift=2*UP)
        self.add(titre)

        subTitle = Text("Phase de sauvegarde").scale(.35).shift(1.5*UP)
        titre.add(subTitle)
        self.play(FadeIn(subTitle))
        self.wait(1)

        text1 = TextItem("Traitement ultérieur des données")
        text2 = TextItem("Permet de personnaliser l'affichage").shift(DOWN)
        text3 = TextItem("Lourd en mémoire, format LZMA lent en compression mais rapide en décompression").shift(2*DOWN)

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
        gradDescr = Text("Environnement Swimmer, algorithme SAC de 250 à 10.000 pas").scale(.2).shift(2*DOWN)
        gradGroup = Group(grad1, grad2, grad3, gradDescr)

        vign1 = ImageMobject("Images/vignette2D2.png").scale(scaleVignette).shift(2*UP)
        vign2 = ImageMobject("Images/vignette2D1.png").scale(scaleVignette)
        vign3 = ImageMobject("Images/vignette2D3.png").scale(scaleVignette).shift(2*DOWN)
        vignDescr = Text("Environnement Pendulum, algorithme SAC à 5000 pas").scale(.2).shift(3.2*DOWN)
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
