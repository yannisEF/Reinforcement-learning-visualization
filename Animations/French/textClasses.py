from manim import *

class SplashScreen(Group):
    def __init__(self, title="", titleScale=1, titleShift=[0,0,0], imgPath=None, imgScale=1, imgShift=[0,0,0]):
        self.title = Text(title).scale(titleScale).shift(titleShift)

        if imgPath is not None:
            self.image = ImageMobject(imgPath).scale(imgScale).shift(imgShift)
            super().__init__(self.title, self.image)
        else:
            super().__init__(self.title)

class TextItem(VGroup):
    def __init__(self, text="", textScale=.5):
        self.text = Text(text).scale(textScale)
        self.text.shift(.6 * RIGHT - self.text[0].get_center())
        self.dot = Dot()

        super().__init__(self.text, self.dot)