from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile

from model.py import Model




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device
    print("Loading model")
    model = Model()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    yield

    print("Cleaning up")
    del model, device

app = FastAPI(lifespan=lifespan)

# IDEA ?? maybe use this for showing card + predicted rank & suit
# @app.post("/caption/")
# async def caption(data: UploadFile = File(...)):
#     """Generate a caption for an image."""
#     i_image = Image.open(data.file)
#     if i_image.mode != "RGB":
#         i_image = i_image.convert(mode="RGB")

#     card_values = card_suit, card_rank
#     pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     return [pred.strip() for pred in preds]