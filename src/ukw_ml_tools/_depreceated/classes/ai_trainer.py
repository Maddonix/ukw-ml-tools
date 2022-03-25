from .db import Db

class AiTrainer:
    """[summary]
    """
    
    def __init__(self, ai_name: str, db: Db):
        self.ai_name = ai_name
        self.db = db
        self.db_images = db.db_images
        self.db_interventions = db.db_interventions
        self.cfg = db.cfg
        self.specs = self.cfg["models"][ai_name]


    def train_ai(self):
        # orig_cwd = hydra.utils.get_original_cwd()
        # configuration for folder structure during model generation in: hydra/default.yaml
        pass
