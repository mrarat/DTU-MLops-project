from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
