from bson import ObjectId
from pydantic import (
    BaseModel, 
    validator
)
from typing import Union

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    def __hash__(self):
        return hash(repr(self))

    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     field_schema.update(type="string")

class Flank(BaseModel):
    name: str
    value: Union[bool, str]
    start: int
    stop: int

    @validator("name")
    def name_to_lower(cls, v):
        return v.lower()

    def __hash__(self):
        return hash(repr(self))


class SettingsSmoothing(BaseModel):
    pass
