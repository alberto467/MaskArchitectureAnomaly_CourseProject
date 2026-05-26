# COCO Classes
# 1: person/person
# 2: vehicle/bicycle
# 3: vehicle/car
# 4: vehicle/motorcycle
# 5: vehicle/airplane
# 6: vehicle/bus
# 7: vehicle/train
# 8: vehicle/truck
# 9: vehicle/boat
# 10: outdoor/traffic light
# 11: outdoor/fire hydrant
# 13: outdoor/stop sign
# 14: outdoor/parking meter
# 15: outdoor/bench
# 16: animal/bird
# 17: animal/cat
# 18: animal/dog
# 19: animal/horse
# 20: animal/sheep
# 21: animal/cow
# 22: animal/elephant
# 23: animal/bear
# 24: animal/zebra
# 25: animal/giraffe
# 27: accessory/backpack
# 28: accessory/umbrella
# 31: accessory/handbag
# 32: accessory/tie
# 33: accessory/suitcase
# 34: sports/frisbee
# 35: sports/skis
# 36: sports/snowboard
# 37: sports/sports ball
# 38: sports/kite
# 39: sports/baseball bat
# 40: sports/baseball glove
# 41: sports/skateboard
# 42: sports/surfboard
# 43: sports/tennis racket
# 44: kitchen/bottle
# 46: kitchen/wine glass
# 47: kitchen/cup
# 48: kitchen/fork
# 49: kitchen/knife
# 50: kitchen/spoon
# 51: kitchen/bowl
# 52: food/banana
# 53: food/apple
# 54: food/sandwich
# 55: food/orange
# 56: food/broccoli
# 57: food/carrot
# 58: food/hot dog
# 59: food/pizza
# 60: food/donut
# 61: food/cake
# 62: furniture/chair
# 63: furniture/couch
# 64: furniture/potted plant
# 65: furniture/bed
# 67: furniture/dining table
# 70: furniture/toilet
# 72: electronic/tv
# 73: electronic/laptop
# 74: electronic/mouse
# 75: electronic/remote
# 76: electronic/keyboard
# 77: electronic/cell phone
# 78: appliance/microwave
# 79: appliance/oven
# 80: appliance/toaster
# 81: appliance/sink
# 82: appliance/refrigerator
# 84: indoor/book
# 85: indoor/clock
# 86: indoor/vase
# 87: indoor/scissors
# 88: indoor/teddy bear
# 89: indoor/hair drier
# 90: indoor/toothbrush
# 92: textile/banner
# 93: textile/blanket
# 95: building/bridge
# 100: raw-material/cardboard
# 107: furniture-stuff/counter
# 109: textile/curtain
# 112: furniture-stuff/door-stuff
# 118: floor/floor-wood
# 119: plant/flower
# 122: food-stuff/fruit
# 125: ground/gravel
# 128: building/house
# 130: furniture-stuff/light
# 133: furniture-stuff/mirror-stuff
# 138: structural/net
# 141: textile/pillow
# 144: ground/platform
# 145: ground/playingfield
# 147: ground/railroad
# 148: water/river
# 149: ground/road
# 151: building/roof
# 154: ground/sand
# 155: water/sea
# 156: furniture-stuff/shelf
# 159: ground/snow
# 161: furniture-stuff/stairs
# 166: building/tent
# 168: textile/towel
# 171: wall/wall-brick
# 175: wall/wall-stone
# 176: wall/wall-tile
# 177: wall/wall-wood
# 178: water/water-other
# 180: window/window-blind
# 181: window/window-other
# 184: plant/tree-merged
# 185: structural/fence-merged
# 186: ceiling/ceiling-merged
# 187: sky/sky-other-merged
# 188: furniture-stuff/cabinet-merged
# 189: furniture-stuff/table-merged
# 190: floor/floor-other-merged
# 191: ground/pavement-merged
# 192: solid/mountain-merged
# 193: plant/grass-merged
# 194: ground/dirt-merged
# 195: raw-material/paper-merged
# 196: food-stuff/food-other-merged
# 197: building/building-other-merged
# 198: solid/rock-merged
# 199: wall/wall-other-merged
# 200: textile/rug-merged

# Cityscapes Classes
# 7: flat/road -> ground/road
# 8: flat/sidewalk -> ground/pavement-merged
# 11: construction/building -> building-other, house, skyscraper, roof
# 12: construction/wall -> wall-brick, wall-stone, wall-tile, wall-wood, wall-other-merged
# 13: construction/fence -> fence, railing
# 17: object/pole
# 19: object/traffic light -> traffic light
# 20: object/traffic sign -> stop sign
# 21: nature/vegetation -> plant/tree-merged, plant/grass-merged, plant/flower
# 22: nature/terrain -> ground/sand, ground/dirt-merged, ground/gravel, ground/platform, ground/playingfield, ground/snow
# 23: sky/sky -> sky-other-merged
# 24: human/person -> person
# 25: human/rider -> person, bicycle, motorcycle
# 26: vehicle/car -> car
# 27: vehicle/truck -> truck
# 28: vehicle/bus -> bus
# 31: vehicle/train -> train
# 32: vehicle/motorcycle -> motorcycle
# 33: vehicle/bicycle -> bicycle

import json

import torch
from datasets.coco_panoptic import CLASS_MAPPING

CATEGORIES_JSON_PATH = "./coco_categories.json"

with open(CATEGORIES_JSON_PATH, 'r') as f:
    categories = json.load(f)

# print("\n".join([ f"{o['id']}: {o['supercategory']}/{o['name']}" for o in categories ]))
coco_classes = [ (CLASS_MAPPING[o['id']], o['name']) for o in categories if o['id'] in CLASS_MAPPING ]
assert len([ o for o in categories if o['id'] not in CLASS_MAPPING ]) == 0, "Some COCO classes are not mapped in CLASS_MAPPING"

from torchvision.datasets import Cityscapes

# print("\n".join([ f"{o.id}: {o.category}/{o.name}" for o in Cityscapes.classes if not o.ignore_in_eval ]))
city_classes = [ (o.train_id, o.name) for o in Cityscapes.classes if not o.ignore_in_eval ]

CITY_FROM_COCO = {
  "road": ["road"],
  "sidewalk": ["pavement-merged"],
  "building": ["building-other-merged", "house", "roof"],
  "wall": ["wall-brick", "wall-stone", "wall-tile", "wall-wood", "wall-other-merged"],
  "fence": ["fence-merged"],
  "pole": [],
  "traffic light": ["traffic light"],
  "traffic sign": ["stop sign"], # Maybe a bit strict?
  "vegetation": ["tree-merged", "grass-merged", "flower"],
  "terrain": ["sand", "dirt-merged", "gravel", "platform", "playingfield", "snow"],
  "sky": ["sky-other-merged"],
  "person": ["person"],
  "rider": ["person", "bicycle", "motorcycle"],
  "car": ["car"],
  "truck": ["truck"],
  "bus": ["bus"],
  "train": ["train"],
  "motorcycle": ["motorcycle"],
  "bicycle": ["bicycle"],
}

missing_city_classes = set(CITY_FROM_COCO.keys()) - set([name for _, name in city_classes])
if len(missing_city_classes) > 0:
    for cls in missing_city_classes:
        print(f"Missing Cityscapes class in mapping: {cls}")
    raise ValueError(f"Some Cityscapes classes are missing in the mapping: {missing_city_classes}")

missing_coco_classes = set([label for labels_list in CITY_FROM_COCO.values() for label in labels_list]) - set([name for _, name in coco_classes
])
if len(missing_coco_classes) > 0:
    for cls in missing_coco_classes:
        print(f"Missing COCO class in mapping: {cls}")
    raise ValueError(f"Some COCO classes are missing in the mapping: {missing_coco_classes}")


COCO_LABEL_TO_IDX = { name: label for label, name in coco_classes }
CITY_LABEL_TO_IDX = { name: label for label, name in city_classes }


def coco_city_class_head_remap(coco_state_dict):
    class_head_prefix = "network.class_head"
    weight_key = f"{class_head_prefix}.weight"
    bias_key = f"{class_head_prefix}.bias"

    if weight_key not in coco_state_dict or bias_key not in coco_state_dict:
        raise KeyError(f"Missing source class head in model")
    
    CITY_CLASSES = 19+1

    old_weights = coco_state_dict[weight_key]
    old_bias = coco_state_dict[bias_key]
    new_weights = torch.zeros(CITY_CLASSES, old_weights.shape[1], device=old_weights.device)
    new_bias = torch.zeros(CITY_CLASSES, device=old_bias.device)

    # Keep last class, which is the "background" class, unchanged
    new_weights[-1] = old_weights[-1]
    new_bias[-1] = old_bias[-1]

    for city_cls, coco_clss in CITY_FROM_COCO.items():
        city_idx = CITY_LABEL_TO_IDX[city_cls]

        if len(coco_clss) > 0:
            for coco_cls in coco_clss:
                coco_idx = COCO_LABEL_TO_IDX[coco_cls]

                new_weights[city_idx] += old_weights[coco_idx]
                new_bias[city_idx] += old_bias[coco_idx]

            new_weights[city_idx] /= len(coco_clss)
            new_bias[city_idx] /= len(coco_clss)
        else:
            new_weights[city_idx] = torch.normal(mean=old_weights.mean(dim=0), std=(old_weights.std(dim=0) + 1e-6) / 10)
            new_bias[city_idx] = torch.normal(mean=old_bias.mean(), std=(old_bias.std() + 1e-6) / 10)

    coco_state_dict[weight_key] = new_weights
    coco_state_dict[bias_key] = new_bias

# Simple class remap with 8 predefined common classes

from torchvision.datasets import Cityscapes
# Debug
#for cls in Cityscapes.classes:
    #print(cls.id, cls.train_id, cls.name, cls.ignore_in_eval)

CITY_CLASS_LABELS = {
    cls.train_id: cls.name
    for cls in Cityscapes.classes
    if not cls.ignore_in_eval
}

COMMON_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light"]
COMMON_LABELS_TO_ID = {name: i for i, name in enumerate(COMMON_CLASSES)}

CITY_TO_COMMON = {
    11: COMMON_LABELS_TO_ID["person"],
    18: COMMON_LABELS_TO_ID["bicycle"],
    13: COMMON_LABELS_TO_ID["car"],
    17: COMMON_LABELS_TO_ID["motorcycle"],
    15: COMMON_LABELS_TO_ID["bus"],
    16: COMMON_LABELS_TO_ID["train"],
    14: COMMON_LABELS_TO_ID["truck"],
    6:  COMMON_LABELS_TO_ID["traffic light"],
}

COMMON_LABELS_TO_COCO = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "traffic light": 10,
}

# We directly use the standard IDs of COCO and the CLASS_MAPPING defined in coco_panoptic.py
import importlib
coco_module = importlib.import_module("datasets.coco_panoptic")
COCO_CLASS_MAPPING = coco_module.CLASS_MAPPING

COCO_TO_COMMON = {
    COCO_CLASS_MAPPING[orig_id]: COMMON_LABELS_TO_ID[name]
    for name, orig_id in COMMON_LABELS_TO_COCO.items()
}

COMMON_TO_COCO = {common_id: coco_id for coco_id, common_id in COCO_TO_COMMON.items()}



# More advanced remap using above mapping
mask_remap_dict = {}
for city_cls, coco_clss in CITY_FROM_COCO.items():
    for coco_cls in coco_clss:
        if coco_cls not in mask_remap_dict:
            mask_remap_dict[coco_cls] = { 'to': city_cls, 'specificity': len(coco_clss) }
        elif mask_remap_dict[coco_cls]['specificity'] > len(coco_clss):
            mask_remap_dict[coco_cls] = { 'to': city_cls, 'specificity': len(coco_clss) }
mask_remap_dict = { COCO_LABEL_TO_IDX[coco_cls]: CITY_LABEL_TO_IDX[info['to']] for coco_cls, info in mask_remap_dict.items() }

unique_city_mapped = set(mask_remap_dict.values())

CITY_TO_COMMON_EXP = { city_id: i for i, city_id in enumerate(unique_city_mapped) }
COCO_TO_COMMON_EXP = { coco_id: CITY_TO_COMMON_EXP[city_id] for coco_id, city_id in mask_remap_dict.items() }
COMMON_CLASSES_EXP = [ CITY_CLASS_LABELS[city_id] for city_id in unique_city_mapped ]