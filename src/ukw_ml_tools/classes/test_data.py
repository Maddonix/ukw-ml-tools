from .base import PyObjectId
from bson import ObjectId
from pydantic import BaseModel, Field
from typing import List


class DbTestSet(BaseModel):
    id: PyObjectId = Field(alias="_id")
    name: str
    ids: List[PyObjectId]
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {}
        }
