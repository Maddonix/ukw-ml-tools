from ..mongodb.get_objects import get_intervention
from typing import Tuple, List
from itertools import groupby
from operator import itemgetter
from ..classes.base import Flank
from collections import defaultdict
from ..classes.annotation import VideoSegmentationAnnotation
import numpy as np
from sklearn.preprocessing import label_binarize

def get_consecutive_ranges(list) -> List[Tuple]:
    range_tuples = []
    for k, g in groupby(enumerate(list), lambda x: x[0]-x[1]):
        range_tuples.append([_ for _ in map(itemgetter(1), g)])

    for i in range(len(range_tuples)):
        range_tuples[i] = (range_tuples[i][0], range_tuples[i][-1]+1)

    return range_tuples

def range_tuples_to_default_flanks(name, value, range_tuples):
    flanks = [Flank(name=name, value=value, start=_[0], stop=_[1]) for _ in range_tuples]
    return flanks

def get_default_label_flanks(flanks: List[Flank], lookup, video_key, db_interventions):
    intervention = get_intervention(video_key, db_interventions)
    n_frames = intervention.metadata.frames_total

    flank_dict = {_: [] for _ in lookup.keys()}
    for flank in flanks:
        flank_dict[flank.name].extend([i for i in range(flank.start, flank.stop)])
    
    add_defaults = defaultdict(list)
    for key in flank_dict.keys():
        flank_dict[key]=set(flank_dict[key])
        for i in range(n_frames):
            if i not in flank_dict[key]:
                add_defaults[key].append(i)
    
    new_flanks = []
    for key in add_defaults.keys():
        default_value = lookup[key]
        if default_value != None:
            add_defaults[key].sort()
            ranges = get_consecutive_ranges(add_defaults[key])
            new_flanks.extend(range_tuples_to_default_flanks(key, lookup[key], ranges))   

    return new_flanks

def binarize_multiclass_array(pred: np.array, choices: List):
    classes = [i for i in range(len(choices))]
    binarized_predictions = label_binarize(pred, classes = classes)
    return binarized_predictions

