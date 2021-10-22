from typing import List
import requests
import xml.etree.ElementTree as ET
import json
from bson.objectid import ObjectId


def delete_project_by_id(project_id: int, labelstudio_prefix: str, labelstudio_token: str):
    requests.delete(f"{labelstudio_prefix}projects/{project_id}", headers={"Authorization": labelstudio_token})


# db -> Labelstudio
def generate_ls_project(label, description, path_label_configs, labelstudio_prefix, labelstudio_token):

    label_config_path = path_label_configs.joinpath(label).with_suffix(".xml")
    assert label_config_path.exists()

    label_config = ET.parse(label_config_path)
    label_config = label_config.getroot()
    label_config = ET.tostring(label_config)

    payload = {
        "title": label,
        "description": description,
        "label_config": label_config,
        "expert_instruction": None,
        "show_instruction": False,
        "show_skip_button": True,
        "enable_empty_annotation": True,
        "show_annotation_history": True,
        "organization": None,
        "color": None,
        "maximum_annotations": 1,
        "is_published": True,
        "model_version": None,
        "sampling": "Uniform sampling",

    }

    r = requests.post(f"{labelstudio_prefix}projects/", headers={"Authorization": labelstudio_token}, data=payload)
    assert r.status_code == 201
    return r.json()


def db_images_to_ls_project(images, label, description, path_label_configs, labelstudio_prefix, labelstudio_token, fileserver_prefix):
    new_project = generate_ls_project(label, description, path_label_configs, labelstudio_prefix, labelstudio_token)
    new_project_id = new_project["id"]
    task_dump = [
        generate_ls_task(image, label, new_project_id, fileserver_prefix) for image in images
        if generate_ls_task(image, label, new_project_id, fileserver_prefix)
        ]
    for task in task_dump:
        task = json.dumps(task)
        r = requests.post(
            f"{labelstudio_prefix}tasks/",
            headers={"Authorization": labelstudio_token, 'content-type': 'application/json'},
            data=task
            )
        assert r.status_code == 201


def generate_ls_task(image, label, new_project_id, fileserver_prefix):
    if "/extreme_storage/files/" in image["path"]:
        task = {
            "data": {
                "image": image["path"].replace("/extreme_storage/files/", fileserver_prefix),
                "prediction": str(image["predictions"][label]["value"]),
                "_id": str(image["_id"])
            },
            "project": str(new_project_id)
        }
        return task


# Labelstudio -> db
def import_ls_project_annotations(project_id, labelstudio_prefix: str, labelstudio_token: str, db_images, verbose: bool = False) -> bool:
    project = requests.get(f"{labelstudio_prefix}projects/{project_id}", headers={"Authorization": labelstudio_token}).json()
    project_labels = project["parsed_label_config"]["choice"]["labels"]

    r = requests.get(
        f"{labelstudio_prefix}projects/{project_id}/tasks/",
        headers={"Authorization": labelstudio_token},
        params={"page_size": "-1"}
        )
    project_tasks = r.json()
    project_tasks_labeled = [_ for _ in project_tasks if _["is_labeled"] is True and _["annotations"]]

    if verbose:
        print(f"Project ID: {project_id}")
        print("Annotated Labels:")
        print(project_labels)
        print(f"Found {len(project_tasks)} Tasks")
        print(f"Processing {len(project_tasks_labeled)} labeled Tasks")
        print("\n")

    with open(f"./ls_dumps/{project_id}.json", "w") as f:
        json.dump(project_tasks, f)
    with open(f"./ls_dumps/{project['id']}_{project['title']}_meta.json", "w") as f:
        json.dump(project, f)

    project_updates = {}
    for task in project_tasks_labeled:
        db_id = task["data"]["_id"]

        project_updates[db_id] = labeled_ls_task_to_db_update(task, project_labels)
    for _id, update in project_updates.items():
        db_images.update_one({"_id": ObjectId(_id)}, {"$set": update})

    return True


def ls_choices_to_label_dict(ls_results: List[dict], label_list: List[str], prefix: str = "labels_new"):
    """Expects labelstudio result (task["annotation"][n_annotation"]["result"]).\
        Asserts that result is of type choices. Reads result and returns label dict.

    Args:
        ls_results (List[dict]): results of ls annotation
        label_list (List[str]): List of all labels to choose from in this project.
        prefix (str, optional): Prefix reffering to field in which the labels are stored\
            in mongo db. Defaults to "labels_new".

    Returns:
        dict: dict of form {prefix.label: bool}
    """
    if ls_results:
        assert len(ls_results) == 1
        result = ls_results[-1]
        assert result["type"] == "choices"
        choices = result["value"]["choices"]
        if "unclear" in choices:
            prefix = "labels_unclear"
        label_dict = {}
        for _ in label_list:
            label_dict[f"{prefix}.{_}"] = False

        for choice in choices:
            label_dict[f"{prefix}.{choice}"] = True
    else:
        label_dict = {}
        for _ in label_list:
            label_dict[f"{prefix}.{_}"] = False

    return label_dict


def labeled_ls_task_to_db_update(task: dict, project_labels: List):
    """Expects a task exported from labelstudio. Task must have at least one annotation result.\
        Lates annotation will be passed to ls_choices_to_label_dict.

    Args:
        task (dict): Labelstudio task with at least one annotation
        project_labels (List): List of all labels available in this project.\
            If not contained in annotation, label will be False.

    Returns:
        dict: Dictionary containig the update.
    """
    assert "annotations" in task
    assert task["annotations"]

    annotations = task["annotations"]
    latest_result = annotations[-1]["result"]
    update_label_dict = ls_choices_to_label_dict(latest_result, project_labels)

    return update_label_dict
