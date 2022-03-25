from ..classes.intervention import Intervention
from collections import defaultdict
from ..classes.annotation import VideoSegmentationAnnotation
from datetime import datetime as dt
from ..classes.polyp import Polyp


def polyps_to_instances(intervention:Intervention, db_interventions, db_images, db_instances):
    select = ["polyp_1", "polyp_2", "polyp_3"]
    select = select + ["new_"+_ for _ in select]
    name = "polyp"
    flanks = []
    any_polyps = False

    for _ in intervention.video_segments_annotation.keys():
        if _ in select:
            any_polyps = True
            flanks.extend(intervention.video_segments_annotation[_].value)
            annotator = intervention.video_segments_annotation[_].annotator_id

    if any_polyps:

        flanks.sort(key = lambda x: x.start, reverse = False)
        polyp_instances = defaultdict(list)
        instance_id = 0
        label = "polyp"

        existing_polyps = [_ for _ in db_instances.find({"intervention_id": intervention.id, "name": "polyp"})]

        active = {
            1: False,
            2: False,
            3: False
        }
        _flanks =[]
        for flank in flanks:
            flank = flank.copy()
            p_label = int(flank.name[-1])
            
            if "new" in flank.name:
                active[p_label] = False
            else:
                status = active[p_label]
                if not status:
                    instance_id += 1
                    active[p_label] = instance_id
                    status = instance_id

                flank.name = label
                _flanks.append(flank)
                polyp_instances[status].append(flank)

        if existing_polyps:
            assert len(existing_polyps) == len(polyp_instances)

        all_polyps = VideoSegmentationAnnotation(
            value = _flanks,
            source = "video_segmentation",
            annotator_id = annotator,
            date = dt.now(),
            name = label
        )
        intervention.video_segments_annotation["polyp"] = all_polyps

        db_interventions.update_one(
            {"intervention_id": intervention.id},
            {"$set": {"video_segments_annotation.polyp": all_polyps.dict()}})

        for key, value in polyp_instances.items():
            frames = []
            for flank in value:
                frames.extend([_ for _ in range(flank.start, flank.stop)])

            image_ids = {
                _["metadata"]["frame_number"]: _["_id"] 
                for _ in db_images.find({"intervention_id": intervention.id, "metadata.frame_number": {"$in": frames}})
                }

            polyp = Polyp(
                intervention_id = intervention.id,
                video_key = intervention.video_key,
                images = image_ids,
                intervention_instance = key
            )

            db_instances.update_one(
                {"intervention_id": intervention.id, "intervention_instance": key},
                {"$set": polyp.to_dict()}, upsert=True)

    return intervention
        