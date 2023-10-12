import argparse
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

from pipeline import run_pipeline


def vqatask(image, question):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_image = Image.open(image).convert('RGB')

    model, vis_processors, txt_processors = (
        load_model_and_preprocess(name='blip_vqa', model_type='vqav2', is_eval=True, device=device))
    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    question = txt_processors['eval'](question)
    return model.predict_answers(samples={'image': image, 'text_input': question}, inference_method='generate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ds', type=str, required=True, help='Path to dataset')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    args = parser.parse_args()

    run_pipeline(vqatask, args.path_to_ds, limit=args.limit)
