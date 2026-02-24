import torch
from huggingface_hub import snapshot_download

def load_model():

    model_path = snapshot_download(
        repo_id="Lightricks/LTX-Video-2",
        local_dir="./model"
    )

    model = torch.load(f"{model_path}/model.pt")

    model.eval()

    return model


def generate_video(model, prompt):

    video = model.generate(prompt)

    path = "output.mp4"

    video.save(path)

    return path
