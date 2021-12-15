from typing import Collection
from bson.objectid import ObjectId
from pymongo.collection import Collection
from .utils import delete_frame_from_intervention

class Image:
    """asd
    asd
    """
    def __init__(self, object_id: ObjectId, db_images: Collection, db_interventions: Collection):
        self.db_images = db_images
        self.db_interventions = db_interventions
        self.image = self.db_images.find_one({"_id": object_id})
        assert self.image
        self.n_frame = self.image["n_frame"]
        self._id = self.image["_id"]

    def delete(self):
        _ = delete_frame_from_intervention(
            self.image,
            self.db_interventions
        )

        self.db_images.delete_one({"_id": self._id})
        return True

    
