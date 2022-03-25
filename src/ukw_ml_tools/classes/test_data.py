from typing import List
from typing import Optional

from bson import ObjectId
from pydantic import BaseModel
from pydantic import Field

from .base import PyObjectId


class DbTestSet(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    name: str
    ids: List[PyObjectId]

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {}
        }
