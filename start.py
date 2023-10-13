import argparse
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

from pipeline import run_pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
blip_model = None


def load_model():
    global blip_model
    blip_model = load_model_and_preprocess(name='blip_vqa', model_type='vqav2', is_eval=True, device=device)


def vqatask(image, question):
    if blip_model is None:
        load_model()
    model, vis_processors, txt_processors = blip_model
    raw_image = Image.open(image).convert('RGB')
    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    question = txt_processors['eval'](question)
    return model.predict_answers(samples={'image': image, 'text_input': question}, inference_method='generate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ds', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Path to output')
    parser.add_argument('--split', type=str, default='train', help='Set to "train" or "test"')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    args = parser.parse_args()

    run_pipeline(vqatask, args.path_to_ds, args.output_dir_name, limit=args.limit, start_at=args.start_at)
