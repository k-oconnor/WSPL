"""Preparation-only spatial fixtures; inference uses only the string codec."""
import random,hashlib


def normal(shape):
    return tuple(sorted((r-min(x for x,y in shape),c-min(y for x,y in shape)) for r,c in shape))


def orientations(shape):
    result=set()
    for flip in (-1,1):
        s=tuple((r,flip*c) for r,c in shape)
        for _ in range(4):result.add(normal(s));s=tuple((c,-r) for r,c in s)
    return sorted(result)


def canonical(shape):return min(orientations(shape))


def families(n):
    shapes={((0,0),)}
    for _ in range(1,n):
        nxt=set()
        for s in shapes:
            for r,c in s:
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    p=(r+dr,c+dc)
                    if p not in s:nxt.add(canonical(s+(p,)))
        shapes=nxt
    return sorted(shapes)


def shape_id(shape):return ';'.join(f'{r},{c}' for r,c in canonical(shape))


def movement(rng,shapes,two=False):
    family=rng.choice(shapes);shape=rng.choice(orientations(family));height=max(r for r,c in shape)+1;width=max(c for r,c in shape)+1
    if height>6 or width>6:return None
    top=rng.randrange(7-height);left=rng.randrange(7-width);cells={(r+top,c+left) for r,c in shape}
    r0,r1=top,top+height-1;c0,c1=left,left+width-1;candidates=[]
    for mr in range(6):
        for mc in range(6):
            d=[]
            if c0<=mc<=c1 and mr<r0:d.append(('up',-1,0))
            if c0<=mc<=c1 and mr>r1:d.append(('down',1,0))
            if r0<=mr<=r1 and mc<c0:d.append(('left',0,-1))
            if r0<=mr<=r1 and mc>c1:d.append(('right',0,1))
            if len(d)!=1:continue
            direction,dr,dc=d[0];end={(r+(2 if two else 1)*dr,c+(2 if two else 1)*dc) for r,c in cells}
            once={(r+dr,c+dc) for r,c in cells}
            if (mr,mc) in end|once or any(not(0<=r<6 and 0<=c<6) for r,c in end):continue
            candidates.append((mr,mc,direction,once,end))
    if not candidates:return None
    mr,mc,direction,once,end=rng.choice(candidates);color=rng.randrange(1,9);a=[0]*36;b=[0]*36;c=[0]*36
    a[mr*6+mc]=b[mr*6+mc]=c[mr*6+mc]=9
    for r,k in cells:a[r*6+k]=color
    for r,k in once:b[r*6+k]=color
    for r,k in end:c[r*6+k]=color
    return dict(size=6,task='move',input=a,target=b,second_target=c if two else None,shape=shape_id(family),color=color,direction=direction)


def prepare(seed=6101,excluded=()):
    rng=random.Random(seed);f4=families(4)
    train_shapes=families(1)+families(2)+families(3)+f4[:3]
    novel_shapes=f4[3:]+families(5);larger_shapes=families(6)
    seen={(len(x),tuple(x)) for x in excluded}
    def take(n,shapes,two=False):
        rows=[];attempts=0
        while len(rows)<n:
            attempts+=1
            if attempts>n*1000:raise ValueError('fixture pool exhausted')
            row=movement(rng,shapes,two)
            if row is None:continue
            key=(len(row['input']),tuple(row['input']))
            if key in seen:continue
            seen.add(key);row['id']=hashlib.sha256(bytes(row['input'])).hexdigest();rows.append(row)
        return rows
    def copies(n,size):
        rows=[]
        while len(rows)<n:
            if size==4:grid=[rng.randrange(4) for _ in range(16)]
            elif len(rows)%2:
                grid=[0]*36
                for i in rng.sample(range(36),rng.randint(2,10)):grid[i]=rng.randrange(1,10)
            else:grid=[rng.randrange(10) for _ in range(36)]
            key=(len(grid),tuple(grid))
            if key in seen:continue
            seen.add(key);rows.append(dict(size=size,task='copy',input=grid,target=grid.copy(),id=hashlib.sha256(bytes(grid)).hexdigest()))
        return rows
    train=dict(move=take(8192,train_shapes),copy4=copies(4096,4),copy6=copies(4096,6))
    dev=dict(move=take(512,train_shapes),copy4=copies(256,4),copy6=copies(256,6))
    test=dict(familiar=take(512,train_shapes),novel=take(512,novel_shapes),larger=take(512,larger_shapes),
              copy4=copies(512,4),copy6=copies(512,6),composition=take(256,novel_shapes,two=True))
    return dict(train=train,dev=dev,test=test,shape_families={k:[shape_id(s) for s in v] for k,v in [('train',train_shapes),('novel',novel_shapes),('larger',larger_shapes)]})
