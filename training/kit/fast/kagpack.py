"""Pack a 720-step JSON tape into the flat int32 buffer kagsim expects."""
import json, numpy as np
OPS = ["PASS","NORTH","SOUTH","EAST","WEST","PICKUP","DROP","PLACE",
       "PLANT","WATER","HARVEST","FERTILIZE","DIG",
       "BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE"]
ITEMS = ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL",
         "FERTILIZER","GOOSE","COW","SHEEP"]
MOPS = ["NONE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"]
OPI={o:i for i,o in enumerate(OPS)}; ITI={o:i for i,o in enumerate(ITEMS)}; MOPI={o:i for i,o in enumerate(MOPS)}
def _u(a):
    if not isinstance(a,list) or not a: return (0,0,1)
    op=OPI.get(a[0],len(OPS))
    arg=ITI.get(a[1],255) if len(a)>=2 and isinstance(a[1],str) else 0
    try: n=int(a[2]) if len(a)>=3 else 1
    except Exception: n=1
    return (op,arg,n)
def _o(o):
    if not isinstance(o,list) or not o: return (0,0,0)
    op=MOPI.get(o[0],0)
    if op in (1,2): return (op,0,1)
    if len(o)<3: return (0,0,0)
    return (op,ITI.get(o[1],255),int(o[2]))
def pack(tape):
    buf=[]
    for t in tape:
        units=[t.get("farmer") or ["PASS"]]+list(t.get("hands") or [])
        orders=list(t.get("market") or [])
        buf.append(len(units))
        for u in units: buf.extend(_u(u))
        buf.append(len(orders))
        for o in orders: buf.extend(_o(o))
    return np.asarray(buf,dtype=np.int32), len(tape)
def load(path):
    return pack(json.load(open(path)))
