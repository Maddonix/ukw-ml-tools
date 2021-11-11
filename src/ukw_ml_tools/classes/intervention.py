from typing import Collection
from bson.objectid import ObjectId
from pymongo.collection import Collection
from tqdm import tqdm
from ukw_ml_tools.classes.fieldnames import FIELDNAME_VIDEO_KEY, FIELDNAME_VIDEO_PATH
from ukw_ml_tools.db.crud import delete_frames_from_db
from .utils import *
from .fieldnames import *
import cv2

class Intervention:
    """asd
    asd
    """
    def __init__(self, object_id: ObjectId, db_images: Collection, db_interventions: Collection, cfg: dict):
        self.db_images = db_images
        self.db_interventions = db_interventions
        self.cfg = cfg
        self.base_path_frames = Path(cfg["base_path_frames"])
        self.intervention = self.db_interventions.find_one({"_id": object_id})
        assert self.intervention
        self._id = self.intervention["_id"]
        self.frame_dir = self.base_path_frames.joinpath(self.intervention[FIELDNAME_VIDEO_KEY])
        self.skip_frame_factor = cfg["skip_frame_factor"]


    def get_unique_frame_labels(self) -> List[str]:
        frame_ids = [_id for n_frame, _id in self.intervention[FIELDNAME_FRAMES].items()]
        images = self.db_images.find(
            {
                "_id": {"$in": frame_ids},
                FIELDNAME_LABELS: {"$exists": True, "$nin": [{}]},
                })

        unique_labels = []
        for image in images:
            unique_labels.extend(list(image[FIELDNAME_LABELS].keys()))
        
        unique_labels = list(set(unique_labels))

        return unique_labels

    def get_frame_id(self, n_frame: int):
        return self.intervention[FIELDNAME_FRAMES][str(n_frame)]


    def refresh(self):
        self.intervention = self.db_interventions.find_one({"_id": self._id})
        return self.intervention


    def add_frame_labels(self, frame_labels: dict):
        imported_ids = self.extract_frames([_ for _ in frame_labels.keys()])

        for n_frame, _label in frame_labels.items():
            update_dict = self.frame_label_dict_to_update_dict(_label)
            self.db_images.update_one({"_id": self.get_frame_id(n_frame)}, {"$set": update_dict})

        self.refresh()

        return self.intervention

    def frame_label_dict_to_update_dict(self, frame_label_dict: dict):
        update_dict = {}
        for label, value in frame_label_dict.items():
            update_dict[f"{FIELDNAME_LABELS}.{label}"] = value

        return update_dict

    def get_frame_list(self, as_list: bool = True) -> List[dict]:
        frame_ids = [_id for n_frame, _id in self.intervention[FIELDNAME_FRAMES].items()]
        frame_cursor = self.db_images.find({"_id": {"$in": frame_ids}}) 
        if as_list:
            frame_list = [_ for _ in frame_cursor]
            return frame_list
        else:
            return frame_cursor


    def extract_frames(self, frame_list: List[int] = None, frame_suffix: str = ".png"):
        cap = cv2.VideoCapture(self.intervention[FIELDNAME_VIDEO_PATH])
        if not frame_list:
            frames_total = get_frames_total(cap)
            frame_list = [_ for _ in range(frames_total)]
            
        
        frame_list = [n_frame for n_frame in frame_list if str(n_frame) not in self.intervention[FIELDNAME_FRAMES]]
        frame_list.sort()
        # frame_dicts = []

        for n_frame in tqdm(frame_list):
            frame_path = self.frame_dir.joinpath(f"{n_frame}{frame_suffix}")
            if not self.frame_dir.exists():
                os.mkdir(self.frame_dir)

            cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
            success, image = cap.read()
            assert success

            assert cv2.imwrite(frame_path.as_posix(), image)

            db_image_entry = get_image_db_template(
                origin=self.intervention[FIELDNAME_ORIGIN],
                intervention_id=self.intervention["_id"],
                path=frame_path.as_posix(),
                n_frame=n_frame,
                image_type= IMAGETYPE_FRAME
            )
            self.db_images.insert(db_image_entry)
            # frame_dicts.append(db_image_entry)

        cap.release()
        if frame_list:
            frames = self.db_images.find({FIELDNAME_INTERVENTION_ID: self._id})
            update = {
                FIELDNAME_FRAMES: {
                    str(frame[FIELDNAME_FRAME_NUMBER]): frame["_id"]
                    } for frame in frames
                }

            self.db_interventions.update_one({"_id": self._id}, {"$set": update})
            self.intervention = self.refresh()

            return frames

    def delete_frames(self, frame_list: List[int]):
        frame_ids = [self.intervention[FIELDNAME_FRAMES][str(n_frame)] for n_frame in frame_list]
        frames = self.db_images.find({"_id": {"$in": frame_ids}})
        for frame in frames:
            os.remove(frame[FIELDNAME_IMAGE_PATH])
        self.db_images.delete_many({"_id": {"$in": frame_ids}})
        self.db_interventions.update_one(
            {
                "_id": self._id
            },
            {
                "$unset": {f"{FIELDNAME_FRAMES}.{n_frame}": "" for n_frame in frame_list}
            }              
        )

        self.refresh()


    # def extract_all_frames(self, skip_frame_factor = None):
    #     if not skip_frame_factor:
    #         skip_frame_factor = self.skip_frame_factor


    # def get_fps(self):
    #     pass

    # def get_total_frames(self):
    #     pass
