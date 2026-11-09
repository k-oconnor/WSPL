"""Small designer-authored family, with word and situation answer formats."""
MECHANISMS=('conduction','convection','radiation','evaporation')
SCENARIOS=(
 ('A cold steel clamp warms where it touches a heated copper pipe.',
  'Heated water rises in a tank while cooler water sinks, carrying heat around.',
  'An infrared lamp warms a surface across an evacuated chamber.',
  'Liquid alcohol on a cloth becomes vapor and enters the air.'),
 ('A chilled coin warms where it rests against a hot solid plate.',
  'Warm air rises through a room while cooler air flows downward, transporting heat.',
  'A sensor absorbs infrared energy across an empty space without touching the source.',
  'Liquid water in a shallow tray gradually changes into vapor at its surface.'),
 ('Heat travels through a solid iron bar from its heated tip toward its cool end.',
  'Hot oil rises and cooler oil sinks in a vessel, moving thermal energy with the fluid.',
  'A spacecraft surface is warmed by electromagnetic waves arriving across a vacuum.',
  'Liquid solvent on a glass slide escapes into the air as gas.')
)


def family(index):
    labels=['A','B','C','D'];scenes=SCENARIOS[index];rows=[]
    for i,name in enumerate(MECHANISMS):
        rows.append(dict(id=f'joint_family_{index}_word_{name}',question=scenes[i]+' Which process is described?',
                         choices=dict(label=labels,text=list(MECHANISMS)),answerKey=labels[i],group='word_'+name))
        rows.append(dict(id=f'joint_family_{index}_situation_{name}',question=f'Which situation is an example of {name}?',
                         choices=dict(label=labels,text=list(scenes)),answerKey=labels[i],group='situation_'+name))
    return rows
