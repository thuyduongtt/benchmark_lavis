import csv
from datetime import datetime
from pathlib import Path

import ijson


def run_pipeline_by_question(task, path_to_dataset, output_dir_name, limit=0, start_at=0, split='train'):
    def init_csv_file():
        if not Path(output_dir_name).exists():
            Path(output_dir_name).mkdir(parents=True)

        timestamp = datetime.now().isoformat()
        csvfile = open(f'{output_dir_name}/result_{timestamp}.csv', 'w', encoding='utf-8')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'image', 'question', 'answer', 'prediction', 'n_hop', 'has_scene_graph', 'split'])
        return csvfile, csvwriter

    csv_file, csv_writer = init_csv_file()

    json_data = stream_data(f'{path_to_dataset}/{split}.json', limit=limit, start_at=start_at)

    i = 0
    for d in json_data:
        i += 1

        if i == 1 or i % 100 == 0:
            print(f"[{i}]: {d['image_id']}")

        # split into smaller CSV file every 1000 records
        if i % 1000 == 0:
            csv_file.close()
            csv_file, csv_writer = init_csv_file()

        local_img_path = f"{split}/{d['image_id']}.jpg"
        img_path = f"{path_to_dataset}/" + local_img_path

        prediction = task(img_path, d)
        # prediction = 'prediction'  # turn off model for pipeline testing

        answers = d['answers']
        csv_writer.writerow([d['image_id'], local_img_path, d['question'], answers,
                             prediction, d['n_hop'], d['has_scene_graph'], split])

    csv_file.close()


'''
n_questions: int
exported_time: datetime
questions: array
    image_id
    image_name
    image_dir
    dataset_name
    question_id
    question
    answers
    answers_scores
    choices
    choice_scores
    property_id
    property_label
    n_hop
    has_scene_graph
'''


def stream_data(path_to_json_file, limit=0, start_at=0):
    i = 0
    with open(path_to_json_file) as f:
        datareader = ijson.items(f, 'questions.item')
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            yield record


def run_pipeline_by_image(task, path_to_dataset, output_dir_name, limit=0, start_at=0, split='train'):
    def init_csv_file():
        if not Path(output_dir_name).exists():
            Path(output_dir_name).mkdir(parents=True)

        timestamp = datetime.now().isoformat()
        csvfile = open(f'{output_dir_name}/result_{timestamp}.csv', 'w', encoding='utf-8')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['id', 'image', 'prediction', 'split'])
        return csvfile, csvwriter

    csv_file, csv_writer = init_csv_file()

    i = 0
    for img_file in Path(f'{path_to_dataset}/{split}').iterdir():
        if img_file.name.startswith('.'):
            continue

        i += 1

        if i == 1 or i % 100 == 0:
            print(f"[{i}]: {img_file.name}")

        # split into smaller CSV file every 1000 records
        if i % 1000 == 0:
            csv_file.close()
            csv_file, csv_writer = init_csv_file()

        local_img_path = f"{split}/{img_file.name}"
        img_path = f"{path_to_dataset}/" + local_img_path

        prediction = task(img_path)

        csv_writer.writerow([img_file.name, local_img_path, prediction, split])

    csv_file.close()