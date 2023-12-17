from fastapi import FastAPI
from pydantic import BaseModel
import torch

model_loaded = torch.jit.load("/Users/kirillkotrikov/Documents/End to end project/end_to_end_project/wine_model_scripted.pt")
model_loaded.eval()


class Item(BaseModel):
    parameter_1: float
    parameter_2: float

app = FastAPI()

@app.post("/wine_data/")

async def prediction(item: Item):
    
    item_dict = item.dict()

    return torch.argmax(model_loaded(torch.tensor([[item.parameter_1,item.parameter_2]], dtype=torch.float32))).item()
    