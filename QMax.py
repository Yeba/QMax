import math
import numpy as np
from typing import List
import sys

class Point:
    def __init__(self, x, y, id, w) -> None:
        self.x: float = x
        self.y: float = y
        self.id: int = id
        self.w: float = w

# QMax
class Qube:
    def __init__(Q, xa: float, xb: float, ya: float, yb: float, a: float, b: float, e: float):
        Q.a, Q.b, Q.e = a, b, e
        Q.xa, Q.xb = xa, xb
        Q.ya, Q.yb = ya, yb
        Q.ni, Q.nj = math.ceil(Q.yb / e) - math.floor(Q.ya / e), math.ceil(Q.xb / e) - math.floor(Q.xa / e)
        Q.w = np.zeros((Q.ni, Q.nj))
        Q.ia = int(np.ceil(Q.ya / Q.e))
        Q.ja = int(np.ceil(Q.xa / Q.e))

    def range(Q, p: Point):  # [[ _i, i_, _j, j_,w ], ]    i-y  j-x
        i = int((p.y - Q.ya) / Q.e) if Q.ya <= p.y <= Q.yb else None
        j = int((p.x - Q.xa) / Q.e) if Q.xa <= p.x <= Q.xb else None
        if Q.ya - Q.b / 2 <= p.y <= Q.yb + Q.b / 2:
            _i = int(np.round((max(Q.ya, p.y - Q.b / 2) - Q.ya) / Q.e))
            i_ = int(np.round((min(Q.yb, p.y + Q.b / 2) - Q.ya) / Q.e))
        else:
            i = None
            _i, i_ = None, None
        if Q.xa - Q.a / 2 <= p.x <= Q.xb + Q.a / 2:
            _j = int(np.round((max(Q.xa, p.x - Q.a / 2) - Q.xa) / Q.e))
            j_ = int(np.round((min(Q.xb, p.x + Q.a / 2) - Q.xa) / Q.e))
        else:
            j = None
            _j, j_ = None, None
        return i, j, _i, i_, _j, j_

    def Update(Q, Pdel: List[Point], Padd: List[Point]):
        for p in Padd:
            i, j, _i, i_, _j, j_ = Q.range(p)
            if _i is not None or _j is not None:
                Q.w[_i:i_, _j:j_] += p.w
        for p in Pdel:
            i, j, _i, i_, _j, j_ = Q.range(p)
            if _i is not None or _j is not None:
                Q.w[_i:i_, _j:j_] -= p.w

    def Query(Q) -> Point:
        i, j = np.unravel_index(np.argmax(Q.w, axis=None), Q.w.shape)
        return Point((j + Q.ja + 0.5) * Q.e, (i + Q.ia + 0.5) * Q.e, -1, Q.w[i, j])

    def memory(Q) -> int:
        m = sys.getsizeof(Q.w)
        return m

# QMaxT version 1
class QubeT(Qube):
    def __init__(Q, xa: float, xb: float, ya: float, yb: float, a: float, b: float, e: float):
        super().__init__(xa, xb, ya, yb, a, b, e)
        Q.w_ = np.zeros_like(Q.w)

    def Update(Q, Ts: List[List[Point]]) :
        for T in Ts:
            Q.w_.fill(0)
            for p in T:
                i, j, _i, i_, _j, j_ = Q.range(p)
                if _i is not None or _j is not None:
                    Q.w_[_i:i_, _j:j_] = p.w
            Q.w += Q.w_

    def memory(Q) -> int:
        m = sys.getsizeof(Q.w) + sys.getsizeof(Q.w_)
        return m

# QMaxT version 2, accelerated
class QubeT2: 
    def __init__(Q, bbox, a: float, b: float, e: float):
        Q.rgx, Q.rgy, Q.e = a, b, e # a-x  b-y
        [Q.xa, Q.xb],[Q.ya, Q.yb]= bbox
        Q.ny, Q.nx = math.ceil(Q.yb / e) - math.floor(Q.ya / e), math.ceil(Q.xb / e) - math.floor(Q.xa / e)
        Q.w = np.zeros((Q.nx, Q.ny)) 
        Q.iys,Q.ixs=int(np.ceil(Q.rgy/e)),int(np.ceil(Q.rgx/e))

    def Update(Q,T:np.ndarray,w):
        xiA:np.ndarray= ((T[:,0]-Q.xa-Q.rgx/2) / Q.e+0.5).astype(int) 
        xiA[xiA<0]=0
        xiB:np.ndarray= ((T[:,0]-Q.xa+Q.rgx/2) / Q.e+0.5).astype(int) 
        xiB[xiB>Q.nx]=Q.nx
        yiA:np.ndarray= ((T[:,1]-Q.ya-Q.rgy/2) / Q.e+0.5).astype(int)
        yiA[yiA<0]=0
        yiB:np.ndarray= ((T[:,1]-Q.ya+Q.rgy/2) / Q.e+0.5).astype(int)
        yiB[yiB>Q.ny]=Q.ny
        ps=np.stack([np.stack([xiA+dx,yiA+dy]).T for dx in range(Q.ixs) for dy in range(Q.ixs)]) # shape:(dxy,len(T),2)
        mask=np.ones((len(ps),len(T)),dtype=bool)
        mask[ps[:,:,0]>=xiB]=False
        mask[ps[:,:,1]>=yiB]=False
        ps=ps[mask].reshape((-1,2))
        Q.w[ps[:,0],ps[:,1]]+=w

    def Query(Q) -> Point:
        i, j = np.unravel_index(np.argmax(Q.w, axis=None), Q.w.shape)
        return Point((j + Q.ja + 0.5) * Q.e, (i + Q.ia + 0.5) * Q.e, -1, Q.w[i, j])
    
    def memory(Q):return sys.getsizeof(Q.w)
