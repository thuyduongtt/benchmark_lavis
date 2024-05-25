import argparse

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import random

from pipeline import run_pipeline_by_question, run_pipeline_by_image

blip_model = None
MODEL_NAME = ''
MODEL_TYPE = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    global blip_model

    blip_model = load_model_and_preprocess(name=MODEL_NAME, model_type=MODEL_TYPE, is_eval=True, device=device)

    # ========== VQA Task
    # Available models for BLIP:
    # name='blip_vqa', model_type='vqav2'
    # name='blip_vqa', model_type='okvqa'
    # name='blip_vqa', model_type='aokvqa'

    # Available models for BLIP-2:
    # name="blip2_opt", model_type="pretrain_opt2.7b"
    # name="blip2_opt", model_type="pretrain_opt6.7b"
    # name="blip2_t5", model_type="pretrain_flant5xl"
    # name="blip2_t5", model_type="pretrain_flant5xxl"

    # Available models for InstructBLIP
    # name="blip2_vicuna_instruct", model="vicuna7b"
    # name="blip2_vicuna_instruct", model="vicuna13b"
    # name="blip2_t5_instruct", model="flant5xl"
    # name="blip2_t5_instruct", model="flant5xxl"

    # ========== Image Captioning Task
    # BLIP
    # name="blip_caption", model_type="base_coco"
    # BLIP-2
    # name="blip2_opt", model_type="caption_coco_opt2.7b"
    # name="blip2_opt", model_type="caption_coco_opt6.7b"
    # name="blip2_t5", model_type="caption_coco_flant5xl"


def vqa_task(image, row_data, choices=None):
    # return f'prediction, {image}, {row_data["question"]}'  # turn off model for pipeline testing

    if blip_model is None:
        load_model()
    model, vis_processors, txt_processors = blip_model
    raw_image = Image.open(image).convert('RGB')
    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

    list_of_choices = []

    if choices is None:
        question = row_data['question']
    else:
        question = row_data['question'] + '\n'
        for ii in range(len(choices)):
            list_of_choices.append({
                'symbol': chr(ii + 65),
                'choice': choices[ii]
            })
        for ii in range(len(list_of_choices)):
            question += f"{list_of_choices[ii]['symbol']}. {list_of_choices[ii]['choice']}\n"

        question += "Answer with the option's letter from the given choices directly."

    # print(question)

    if MODEL_NAME == 'blip_vqa':
        question = txt_processors['eval'](question)
        output = model.predict_answers(samples={'image': image, 'text_input': question}, inference_method='generate')

    elif MODEL_NAME in ["blip2_opt", "blip2_t5"]:
        output = model.generate({"image": image, "prompt": f"Question: {question} Answer:"})

    elif MODEL_NAME in ["blip2_vicuna_instruct", "blip2_t5_instruct"]:
        output = model.generate({"image": image, "prompt": question})

    else:
        output = None

    if choices is not None:
        return f'{output} | {[c["symbol"] + ". " + c["choice"] for c in list_of_choices]}'

    return output


def image_captioning_task(image):
    if blip_model is None:
        load_model()
    model, vis_processors, txt_processors = blip_model
    raw_image = Image.open(image).convert('RGB')
    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

    return model.generate({"image": image})


def test_model():
    from pathlib import Path

    print('===== TEST MODEL =====')
    img = 'img/eiffel.jpg'
    assert Path(img).exists(), f'No image in {img}'
    row_data = {
        'question': 'In which country is this tower located?',
        'choices': ['France', 'Germany', 'Vietnam', 'China'],
        'choice_scores': [1, 0, 0, 0]
    }
    r = vqa_task(img, row_data, True)
    print(f'{img}, {row_data["question"]}, {r}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, default='ReasonVQA', help='Valid input: ReasonVQA, VQAv2, OKVQA, GQA')
    parser.add_argument('--ds_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--img_dir', type=str, default='', help='Path to images')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Path to output')
    parser.add_argument('--split', type=str, default='train', help='Set to "train" or "test"')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--multichoice', action='store_true')
    parser.add_argument('--model_name', type=str, default='blip2_t5_instruct')
    parser.add_argument('--model_type', type=str, default='flant5xxl')
    parser.add_argument('--task', type=str, default='vqa', help='Task name: vqa, image_captioning')
    args = parser.parse_args()

    print(args)

    global MODEL_NAME, MODEL_TYPE
    MODEL_NAME = args.model_name
    MODEL_TYPE = args.model_type

    # test_model()

    if args.task == 'vqa':
        run_pipeline_by_question(vqa_task, args.ds_name, args.ds_dir, args.img_dir, args.output_dir_name, limit=args.limit,
                                 start_at=args.start_at, split=args.split, multichoice=args.multichoice)
    elif args.task == 'image_captioning':
        run_pipeline_by_image(image_captioning_task, args.ds_name, args.ds_dir, args.img_dir, args.output_dir_name, limit=args.limit,
                              start_at=args.start_at, split=args.split)
    else:
        print('Invalid task')


def main_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='blip2_t5', help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='pretrain_flant5xxl', help='Path to dataset')
    args = parser.parse_args()

    global MODEL_NAME, MODEL_TYPE
    MODEL_NAME = args.model_name
    MODEL_TYPE = args.model_type

    image_captioning_task('img/bridge.jpg')


if __name__ == '__main__':
    main()
    # main_test()
