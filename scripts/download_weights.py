import wandb
from dotenv import load_dotenv

load_dotenv()
api = wandb.Api()

artifact = api.artifact("jan05torak/mcdetect/run_yolov8n-baseline-4_20260511_215658_model:v0")

path = artifact.download()
print(path)
