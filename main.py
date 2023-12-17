from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    parameter_1: float
    parameter_2: float

app = FastAPI()

@app.post("/wine_data/")

async def prediction(item: Item):
    
    item_dict = item.dict()

    return item.parameter_1 + item.parameter_2
