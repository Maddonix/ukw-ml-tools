from pydantic import BaseModel, Field
from .base import PyObjectId
from pymongo.collection import Collection

class UpdateOne(BaseModel):
    collection: Collection
    match: dict
    operation: dict

    class Config:
        arbitrary_types_allowed = True

    def get_query(self):
        q = dict(self.match), {dict(self.operation)}
        return q

    def post(self):
        r = self.collection.update_one(self.get_query())