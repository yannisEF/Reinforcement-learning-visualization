import random

from manim import *
from textClasses import *
from utils import *

class Introduction(Scene):
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
                self.remove(dashedLine)

            else:
                self.play(MoveAlongPath(point, line))

        return traceLines  
         
    def construct(self):
        # Authors
        splash = SplashScreen(title="\t\t\t\t\tSorbonne University\n\nMaster's of Compteur Science : 1st year research project", titleScale=.5,
                            imgPath="Images/logo_SU.jpeg", imgScale=.5, imgShift=2*UP)
        authors = Text("Students :\nYannis ELRHARBI-FLEURY\nSarah KERRICHE\nLydia AGUINI").scale(.25).shift(2*DOWN+2*RIGHT)
        teacher = Text("Teacher :\nOlivier SIGAUD").scale(.25).shift(2*DOWN+2*LEFT)
        splash.add(authors, teacher)
    
        self.add(splash)
        self.wait(11)
        self.play(FadeOut(splash))
        self.wait(2)

        # Apprentissage
        #   Empty
        traceLines = [] 
        point = Dot([0,0,0], color=RED, radius=1*DEFAULT_DOT_RADIUS)
        self.add_foreground_mobjects(point)

        self.play(FadeIn(point))
        self.wait(2)
        
        points = [point.get_center()] + makeRandomPoints(15, minX=-2, maxX=2, minY=-2, maxY=2)
        path = makePath(points)
        traceLines += self.moveAlong(point, path)

        #   Image + Title
        image = ImageMobject("Images/Background_perlin.png").scale(1)
        title = Text("Reinforcement learning value landscape visualization").shift(3*UP).scale(.5)
        self.play(FadeIn(image), FadeIn(title))

        points = [points[-1]] + makeRandomPoints(12, minX=-2, maxX=2, minY=-2, maxY=2)
        path = makePath(points)
        traceLines += self.moveAlong(point, path)

        points = [points[-1]] + makeRandomPoints(6, minX=-2, maxX=2, minY=-2, maxY=2)
        path = makePath(points)
        traceLines += self.moveAlong(point, path, trace=True)

        learnGroup = Group(point, image, *traceLines)        
        self.play(learnGroup.animate.shift(2*LEFT))
        self.wait(2)

        # Questions
        remark = TextItem("The learning space is in N dimensions.").shift(UP)
        question1 = TextItem("How to visualize the model's trajectory?").shift(UP)
        question2 = TextItem("How to visualize its surroundings?")

        items = Group(remark, question1, question2).scale(.5).shift(DOWN + LEFT)

        #   Appearing questions
        self.play(FadeIn(remark))
        self.wait(6)
        self.play(remark.animate.shift(2*UP), FadeIn(question1))
        self.wait(1)
        self.play(FadeIn(question2))
        self.wait(5)
        self.play(FadeOut(items))

        # Table of content
        part1 = TextItem("1 - Methods of visualization").shift(1.5*UP)
        part2 = TextItem("2 - Structures of the developed tools").shift(.5*UP)
        part3 = TextItem("3 - Demonstration of use").shift(.5*DOWN)
        part4 = TextItem("4 - Future developments").shift(1.5*DOWN)

        items = Group(part1, part2, part3, part4).scale(.5).shift(LEFT)

        #   Appearing table of content
        self.play(FadeIn(part1))
        self.wait(4)
        self.play(FadeIn(part2))
        self.wait(3)
        self.play(FadeIn(part3))
        self.wait(3)
        self.play(FadeIn(part4))
        self.wait(5)
        self.play(FadeOut(items), FadeOut(title), *[FadeOut(t) for t in traceLines], FadeOut(point), FadeOut(image))