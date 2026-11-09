"""Designer-defined factorial heat-transfer questions; not additional natural ARC."""
import hashlib
CHOICES=dict(label=['A','B','C','D'],text=['convection','radiation','conduction','evaporation'])
MECHANISMS=('conduction','convection','radiation','evaporation')
TEMPLATES={
'conduction':[
 'A hot {solid} touches a cooler {other}. Thermal energy crosses their contact surface. Which process transfers that energy?',
 'Energy travels from a warm {solid} into a cold {other} that is touching it. Which heat-transfer mechanism is responsible?',
 'A cooler {other} warms through direct contact with a heated {solid}. What is this heat-transfer process?',
 'Two solid objects, a warm {solid} and a cooler {other}, are pressed together. Heat flows across their interface. Which process is occurring?',
 'The end of a {other} touching a hot {solid} becomes warmer. Heat passes directly through the touching solids. Which process is this?',
 'A heated {solid} and a cooler {other} exchange energy where they touch, without bulk material motion. Which process transfers this heat?',
 'Heat moves through a solid {solid} into a touching {other}, from the warmer part to the cooler part. What is this mechanism?',
 'A hot {solid} rests against a colder {other}. Energy passes between their particles at the contact. Which heat-transfer process occurs?',
 'After a {other} is placed against a warmer {solid}, energy diffuses across the solid interface. Name the heat-transfer mechanism.',
 'A {other} gains thermal energy from a touching {solid}; neither solid moves as a whole. Which transport process explains this?',
 'A {solid} is warm and a {other} is cool. Pressing them together lets microscopic interactions carry energy across the boundary. Which mechanism describes that transfer?',
 'Consider two stationary solids: a heated {solid} against a colder {other}. Energy spreads through the solid connection rather than being carried by flowing material. What is the process?'],
'convection':[
 'Warm {fluid} rises and cooler {fluid} sinks in a container. Which process carries heat with this circulating fluid?',
 'Heat moves around a vessel because warmer {fluid} flows upward and colder {fluid} flows downward. Which mechanism is this?',
 'A container of {fluid} develops rising warm regions and descending cool regions. Thermal energy travels with the moving fluid. What is this process?',
 'Bulk currents of {fluid} carry energy between warmer and colder parts of a tank. Which heat-transfer process occurs?',
 'Warmed {fluid} moves away from a heater and cooler {fluid} replaces it. Which process transports energy by this motion?',
 'A temperature difference creates circulating currents in {fluid}. Heat is carried by the currents themselves. Which mechanism transfers it?',
 'Heat reaches the top of a container when warm {fluid} rises through cooler {fluid}. Which process describes this fluid transport?',
 'The movement of warm and cool {fluid} around a vessel redistributes thermal energy. What is this heat-transfer mechanism?',
 'Parcels of {fluid} move through a vessel, carrying their thermal energy with them. Name the heat-transfer mechanism associated with that bulk transport.',
 'Differences in temperature drive a circulation in {fluid}; the moving regions transport heat. Which mechanism is involved?',
 'Follow a warm parcel of {fluid} as it rises, displacing colder material that descends. Energy travels with the moving parcel. What is the heat-transfer process?',
 'Consider a vessel of {fluid}: its hot and cold regions exchange locations, transporting energy by their collective movement. Which mechanism describes this?'],
'radiation':[
 'A hot {solid} sends infrared electromagnetic waves toward a cooler {other}. Which process transfers energy in these waves?',
 'Energy from a warm {solid} crosses an empty gap as infrared waves and warms a {other}. Which heat-transfer mechanism is this?',
 'A {other} absorbs electromagnetic waves emitted by a heated {solid}. What process transfers thermal energy across the gap?',
 'Without touching or moving warm fluid between them, a hot {solid} warms a {other} through infrared waves. Which process is responsible?',
 'Infrared light emitted from a hot {solid} carries energy to a nearby {other}. What is this heat-transfer process?',
 'Heat travels from a glowing {solid} to a {other} as electromagnetic energy through a vacuum. Which mechanism allows this?',
 'A heated {solid} emits waves that a colder {other} absorbs. The waves carry energy even through empty space. Which process is this?',
 'A {other} becomes warmer after absorbing infrared emissions from a hot {solid}. Which transfer mechanism carries that energy?',
 'Photons emitted by a warm {solid} are absorbed by a {other} across a gap. Name the heat-transfer mechanism.',
 'A {solid} emits thermal energy that reaches a {other} through a vacuum as light. Which mechanism is involved?',
 'Consider a hot {solid} and a colder {other} separated by empty space. Photons carry energy from the hot object to the cold one. What is the heat-transfer process?',
 'No material flows between a {solid} and a {other}, and they do not touch. Thermal electromagnetic emissions from the hot object supply energy to the cooler one. Which mechanism describes this?'],
'evaporation':[
 'Liquid {fluid} slowly escapes from its surface into the air as vapor below its boiling point. Which process is occurring?',
 'A thin layer of {fluid} dries because molecules leave the liquid surface and become gas. What is this process?',
 'The surface of liquid {fluid} loses molecules to the surrounding air as vapor. Which phase-change process is this?',
 'Liquid {fluid} on a {solid} gradually becomes vapor without boiling. Which process removes the liquid?',
 'A wet {other} dries as its liquid {fluid} changes into gas at the surface. What is this process called?',
 'Some molecules at the surface of {fluid} escape into the gas phase, leaving less liquid behind. Which process is this?',
 'An open container loses liquid {fluid} as vapor enters the air from its surface. Which process explains the loss?',
 'A film of {fluid} on a {solid} disappears as surface molecules become vapor below the boiling temperature. What is this process?',
 'Surface molecules depart a pool of liquid {fluid}, joining the surrounding gas while the liquid is not boiling. Name this phase change.',
 'A {other} coated with {fluid} becomes dry as the surface liquid turns into vapor. Which process is involved?',
 'Consider an exposed layer of liquid {fluid} below its boiling temperature. Individual surface molecules escape into the gas around it. What is the process?',
 'The amount of liquid {fluid} on a {solid} decreases as molecules leave its surface and remain airborne in the vapor phase. Which mechanism describes this change?']}


def build_splits():
    locations={
      'train':['kitchen','workshop','classroom','laboratory','greenhouse','garage','laundry room','boathouse'],
      'development':['cafeteria','machine room','school yard','garden shed'],
      'final':['mountain shelter','rail depot','desert station','island cabin','museum workshop','harbor warehouse','orchard hut','field observatory']}
    solids={'train':['steel plate','copper block','iron rod','aluminum pan'],
            'development':['brass bar','ceramic tile','glass sheet','stone slab'],
            'final':['bronze disc','granite block','nickel strip','clay brick']}
    others={'train':['metal spoon','metal stand','metal clamp','metal probe'],
            'development':['steel cup','copper tray','iron nail','ceramic bowl'],
            'final':['glass bead','stone ring','brass cylinder','ceramic rod']}
    result={}
    for split in locations:
        indices=range(8) if split=='train' else range(8,10) if split=='development' else range(10,12)
        rows=[]
        for i,loc in enumerate(locations[split]):
            for j,solid in enumerate(solids[split]):
                setting=f'{loc}/{solid}';fields=dict(solid=solid,other=others[split][j],fluid=['water','ethanol','water','ethanol'][j])
                for mechanism in MECHANISMS:
                    for template in indices:
                        question=f'At the {loc}: '+TEMPLATES[mechanism][template].format(**fields)
                        uid=hashlib.sha256(question.encode()).hexdigest()[:16]
                        rows.append(dict(id='heat_'+uid,question=question,choices=CHOICES,answerKey=CHOICES['label'][CHOICES['text'].index(mechanism)],category=mechanism,setting=setting,template_id=f'{mechanism}_{template}',split=split))
        result[split]=rows
    return result
