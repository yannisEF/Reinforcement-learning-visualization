#!/usr/bin/env python

# AUTHOR davidgutierrezrubio
#   https://raw.githubusercontent.com/davidgutierrezrubio/manimClasses/master/Recolocate.py

from manim import *

class Recolocate(Homotopy):
    #Clase de animación para recolocar una figura
    def __init__(self,mobject,fromA,fromB,toA,toB,scaling=True,rotate=True,**kwargs):
        mobject.init_state = mobject.copy()

        self.update_function=__class__.get_recolocate_udpate_function(fromA,fromB,toA,toB,scaling,rotate)
        super().__init__(self.update_function,mobject,**kwargs)
    @staticmethod
    def get_recolocate_udpate_function(fromAp,fromBp,toAp,toBp,scaling,rotate):
        fromA=__class__.to_point(fromAp)
        fromB=__class__.to_point(fromBp)
        toA=__class__.to_point(toAp)
        toB=__class__.to_point(toBp)
        v1=fromB-fromA
        v2=toB-toA
        transVector=toA-fromA
        if rotate:
            angle=np.arctan2(v2[1],v2[0])-np.arctan2(v1[1],v1[0])
        else:
            angle=0
        if scaling:
            scala=np.linalg.norm(v2)/np.linalg.norm(v1)
        else:
            scala=1
        return lambda x,y,z,t: (np.array([x,y,z])-fromA).dot(np.array([
            [((1-t)+scala*t)*np.cos(t*angle),((1-t)+scala*t)*np.sin(t*angle),0],
            [-((1-t)+scala*t)*np.sin(t*angle),((1-t)+scala*t)*np.cos(t*angle),0],
            [0,0,0]
            ]))+fromA+t*(toA-fromA)
    def applyRtoObj(self,mob,alpha=1):
        mob.apply_function(lambda p:self.update_function(*p,alpha))
        return mob
    @staticmethod
    def to_point(p):#If given an mobject, returns its center
        if isinstance(p,Mobject):
            return p.get_center()
        else:
            return p
    @staticmethod
    def recolocateObject(mobject,fromA,fromB,toA,toB,scaling=True):
        fu=Recolocate.get_recolocate_udpate_function(fromA,fromB,toA,toB,scaling)
        mobject.apply_function(lambda p:fu(*p,1))

class RecolocateFromToSimple(Recolocate):
    def __init__(self,mobject,fromAp,toAp,angle,**kwargs):
        fromA=__class__.to_point(fromAp)
        toA=__class__.to_point(toAp)
        fromB=fromA+RIGHT
        toB=toA+np.array([np.cos(angle),np.sin(angle),0])
        super().__init__(mobject,fromA,fromB,toA,toB,False,**kwargs)


class RecolocateShiftSimple(Recolocate):
    def __init__(self,mobject,vec,angle,**kwargs):
        fromA=np.array([0,0,0])
        fromB=fromA+RIGHT
        toA=fromA+vec
        toB=toA+np.array([np.cos(angle),np.sin(angle),0])
        super().__init__(mobject,fromA,fromB,toA,toB,False,**kwargs)
