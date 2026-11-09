"""Domain fixtures only. Symbolic movement is never called by model inference."""
import hashlib,random
SHAPES={
 'single':((0,0),), 'horizontal':((0,0),(0,1)), 'vertical':((0,0),(1,0)),
 'L0':((0,0),(0,1),(1,0)), 'L1':((0,0),(0,1),(1,1)),
 'L2':((0,0),(1,0),(1,1)), 'L3':((0,1),(1,0),(1,1))}
DIRECTIONS={'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}


def encode_grid(grid):
    if len(grid)!=16 or any(type(x)!=int or not 0<=x<=3 for x in grid):raise ValueError('16 palette cells required')
    return ''.join(map(str,grid))


def enumerate_domain():
    records=[]
    for name,shape in SHAPES.items():
        for top in range(4-max(r for r,c in shape)):
            for left in range(4-max(c for r,c in shape)):
                cells={(r+top,c+left) for r,c in shape}
                r0,r1=min(r for r,c in cells),max(r for r,c in cells)
                c0,c1=min(c for r,c in cells),max(c for r,c in cells)
                for mr in range(4):
                    for mc in range(4):
                        if (mr,mc) in cells:continue
                        directions=[]
                        if c0<=mc<=c1 and mr<r0:directions.append('up')
                        if c0<=mc<=c1 and mr>r1:directions.append('down')
                        if r0<=mr<=r1 and mc<c0:directions.append('left')
                        if r0<=mr<=r1 and mc>c1:directions.append('right')
                        if len(directions)!=1:continue
                        direction=directions[0];dr,dc=DIRECTIONS[direction];moved={(r+dr,c+dc) for r,c in cells}
                        if (mr,mc) in moved or any(not (0<=r<4 and 0<=c<4) for r,c in moved):continue
                        for color in (1,2):
                            before=[0]*16;after=[0]*16;before[mr*4+mc]=after[mr*4+mc]=3
                            for r,c in cells:before[r*4+c]=color
                            for r,c in moved:after[r*4+c]=color
                            text=encode_grid(before);identifier=hashlib.sha256(text.encode()).hexdigest()[:16]
                            records.append(dict(id='grid_'+identifier,input=before,output=after,shape=name,direction=direction,color=color))
    if len({r['id'] for r in records})!=len(records):raise ValueError('duplicate domain inputs')
    return records


def balanced_take(rows,n,seed):
    if n%8:raise ValueError('multiple of direction/color groups required')
    chosen=[]
    for direction in DIRECTIONS:
        for color in (1,2):
            pool=sorted((r for r in rows if r['direction']==direction and r['color']==color),key=lambda r:r['id'])
            random.Random(f'{seed}:{direction}:{color}').shuffle(pool)
            if len(pool)<n//8:raise ValueError('insufficient fixtures')
            chosen+=pool[:n//8]
    return sorted(chosen,key=lambda r:r['id'])


def prepare_sets():
    rows=enumerate_domain();simple=[r for r in rows if not r['shape'].startswith('L')];novel=[r for r in rows if r['shape'].startswith('L')]
    demos=balanced_take(simple,8,61)
    rest=[r for r in simple if r['id'] not in {x['id'] for x in demos}]
    copy=balanced_take(rest,32,71)+balanced_take(novel,32,72);copy_ids={r['id'] for r in copy}
    movement=balanced_take([r for r in rest if r['id'] not in copy_ids],64,81)
    held=balanced_take([r for r in novel if r['id'] not in copy_ids],64,91)
    return dict(demonstrations=demos,copy=copy,movement_familiar=movement,movement_novel=held)
