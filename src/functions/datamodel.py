from pydantic import BaseModel


class OptimumParameter(BaseModel):
    n_neighbours: int
    edge_rule: str
    resolution: int
    mean: float
