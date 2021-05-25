from manim import *

from textClasses import *

class Conclusion(Scene):
	def construct(self):
		splash = SplashScreen(title="Visualisation du paysage de valeur\n					Conclusion", titleScale=.5)
		authors = Text("Étudiants :\nYannis ELRHARBI-FLEURY\nSarah KERRICHE\nLydia AGUINI").scale(.25).shift(2*DOWN+2*RIGHT)
		teacher = Text("Encadrant :\nOlivier SIGAUD").scale(.25).shift(2*DOWN+2*LEFT)
		splash.add(authors, teacher)
		
		self.add(splash)
		self.wait(45)
