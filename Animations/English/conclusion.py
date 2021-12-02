from manim import *

from textClasses import *

class Conclusion(Scene):
	def construct(self):
		splash = SplashScreen(title="Reinforcement learning value landscape visualization\n\t\t\t\t\t\t\t\tConclusion", titleScale=.5)
		authors = Text("Students :\nYannis ELRHARBI-FLEURY\nSarah KERRICHE\nLydia AGUINI").scale(.25).shift(2*DOWN+2*RIGHT)
		teacher = Text("Teacher :\nOlivier SIGAUD").scale(.25).shift(2*DOWN+2*LEFT)
		splash.add(authors, teacher)
		
		self.add(splash)
		self.wait(45)
