from pydantic import BaseModel
from typing import List, Dict


class VectorParams(BaseModel):
    vector: List[float]
    metadata: Dict
    content: str