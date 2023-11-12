import argparse

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

from pipeline import run_pipeline

blip_model = None
MODEL_NAME = ''
MODEL_TYPE = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    global blip_model

    blip_model = load_model_and_preprocess(name=MODEL_NAME, model_type=MODEL_TYPE, is_eval=True, device=device)

    # Available models for BLIP:
    # name='blip_vqa', model_type='vqav2'
    # name='blip_vqa', model_type='okvqa'
    # name='blip_vqa', model_type='aokvqa'

    # Available models for BLIP-2:
    # name="blip2_opt", model_type="pretrain_opt2.7b"
    # name="blip2_opt", model_type="pretrain_opt6.7b"
    # name="blip2_t5", model_type="pretrain_flant5xl"
    # name="blip2_t5", model_type="pretrain_flant5xxl"


def vqatask(image, question):
    if blip_model is None:
        load_model()
    model, vis_processors, txt_processors = blip_model
    raw_image = Image.open(image).convert('RGB')
    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    # question = txt_processors['eval'](question)
    # return model.predict_answers(samples={'image': image, 'text_input': question}, inference_method='generate')
    return model.generate({"image": image, "prompt": f"Question: {question} Answer:"})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ds', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Path to output')
    parser.add_argument('--split', type=str, default='train', help='Set to "train" or "test"')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--model_name', type=str, default='blip2_t5', help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='pretrain_flant5xxl', help='Path to dataset')
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    MODEL_TYPE = args.model_type

    run_pipeline(vqatask, args.path_to_ds, args.output_dir_name, limit=args.limit, start_at=args.start_at,
                 split=args.split)
