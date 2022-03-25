from collections import defaultdict

def annotations_by_label_group(video_annotations):
    '''
    Returns dict with key = label group and list of annotation objects.
    List is ordered by date, index 0 is most recent
    '''
    
    annotation_dict = defaultdict(list)
    for annotation in video_annotations:
        annotation_dict[annotation.label_group_id].append(annotation)

    for group in annotation_dict.keys():
        annotation_dict[group].sort(key= lambda x: x.date, reverse = True)

    return annotation_dict


def get_labels_with_test_annotation(group_ids, test_label_group_dict):
    labels_with_test_annotation = {}
    for _id in group_ids:
        if _id in test_label_group_dict:
            for key, value in test_label_group_dict[_id].items():
                labels_with_test_annotation[key] = value

    return labels_with_test_annotation


def get_flank_dict(video_annotation_dict, test_label_group_dict):
    group_ids = list(video_annotation_dict.keys())
    lookup = get_labels_with_test_annotation(group_ids, test_label_group_dict)
    flanks = {"test": [], "train": []}
    for _id, annotations in video_annotation_dict.items():
        # annotations is a list ordered by date, starting with most recent
        annotation = annotations[0]
        if _id in test_label_group_dict:
            if _id == 4:
                for annotation in annotations:
                    flanks["test"].extend(annotation.flanks)
            else: 
                flanks["test"].extend(annotation.flanks)
        else:
            flanks["train"].extend([_ for _ in annotation.flanks if _.name not in lookup])
        
    return flanks, lookup
