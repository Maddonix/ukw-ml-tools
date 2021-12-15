from .fieldnames import *
from pathlib import Path
from collections import defaultdict
from typing import List

class FastCatAnnotation:
    """asd
    """
    def __init__(self, path: Path, cfg):
        self.skip_frame_factor = cfg["skip_frame_factor"]
        self.fast_cat_allowed_labels = cfg["fast_cat_allower_labels"]
        self.path = path
        self.flanks = self.read_fast_cat_annotation()
        self.labels = self.flanks_to_labels()
        assert path.exists()

    def read_fastcat_annotation(self, skip_frame_factor: int = None, allowed_label_list: List[str] = None):
        # Validate Metadata
        if not skip_frame_factor:
            skip_frame_factor = self.skip_frame_factor
        if not allowed_label_list:
            allowed_label_list = self.fast_cat_allowed_labels

        for label in self.annotation["metadata"]["classes"]:
            assert label in allowed_label_list

        label_list = [_.split(".")[0] for _ in allowed_label_list]
        label_list = list(set(label_list))

        flanks = {_: {"start": [], "stop": []} for _ in label_list}

        for superframe in self.annotation["superframes"]:
            for frame_annotation in superframe["childFrames"]:
                for label in frame_annotation["classes"]:
                    label, param = label.split(".")
                    flanks[label][param].append(frame_annotation["frameNumber"])
                
        for key, value in flanks.items():
            assert len(value["start"]) is len(value["stop"])
            flanks[key]["start"].sort()
            flanks[key]["stop"].sort()

            _matched = []
            for i, _start in enumerate(flanks[key]["start"]):
                _stop = flanks[key]["stop"][i]
                assert _stop >= _start
                _matched.append((_start, _stop))

            flanks[key] = _matched

        for key, value in flanks.items():
            for i, _flank in enumerate(value):
                if i == 0:
                    continue

                assert _flank[0] > value[i - 1][1]


        flanks["skip_frame_factor"] = skip_frame_factor

        return flanks

    def flanks_to_labels(self, flanks_dict: dict = None):
        if not flanks_dict:
            flanks_dict = self.flanks
        frame_labels = defaultdict(dict)
        skip_frame_factor = flanks_dict["skip_frame_factor"]
        for label, _flanks in flanks_dict.items():
            if label == "skip_frame_factor":
                continue
            for _flank in _flanks:
                for n_frame in range(_flank[0], _flank[1]+1):
                    frame_labels[n_frame*skip_frame_factor][label] = True

        return frame_labels



    