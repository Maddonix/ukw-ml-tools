from pathlib import Path
from ..ls.utils import *

class Labelstudio:
    """[summary]
    """

    def __init__(self, ls_url: str, ls_token: str, ls_config_path: Path, cfg: dict):
        self.url = ls_url
        self.token = ls_token
        self.ls_config_path = ls_config_path
        self.fileserver_prefix = cfg["url_endobox_extreme"] + ":" + cfg["port_fileserver"] + "/"


    def delete_project_by_id(self, project_id: int):
        delete_project_by_id(project_id, self.url, self.token)

    def generate_ls_project(self, label: str, description: str):
        return generate_ls_project(
            label,
            description,
            self.ls_config_path,
            self.url,
            self.token
        )

    def images_to_ls_project(self, images, label, description):
        new_project = self.generate_ls_project(label, description)
        new_project_id = new_project["id"]
        task_dump = [self.generate_ls_task(image, label, new_project_id) for image in images]
        task_dump = [_ for _ in task_dump if _]

        for task in task_dump:
            task = json.dumps(task)
            r = requests.post(
                f"{self.url}tasks/",
                headers={"Authorization": self.token, 'content-type': 'application/json'},
                data=task
                )
            assert r.status_code == 201

    def generate_ls_task(self, image, label, project_id):
        if "/extreme_storage/files/" in image["path"]:
            task = {
                "data": {
                    "image": image["path"].replace("/extreme_storage/files/", self.fileserver_prefix),
                    "prediction": str(image["predictions"][label]["value"]),
                    "_id": str(image["_id"])
                },
                "project": str(project_id)
            }
            return task


    def import_ls_project_annotations(self, project_id, db_images, import_as_validated:bool=False, verbose:bool = False):
        import_ls_project_annotations(
            project_id,
            self.url,
            self.token,
            db_images,
            import_as_validated,
            verbose
            )
