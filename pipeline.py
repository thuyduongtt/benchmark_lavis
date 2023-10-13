import ijson
import csv
from datetime import datetime
from pathlib import Path
import ast


def init_csv_file(output_dir_name):
    if not Path(output_dir_name).exists():
        Path(output_dir_name).mkdir(parents=True)

    timestamp = datetime.now().isoformat()
    csv_file = open(f'{output_dir_name}/result_{timestamp}.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['id', 'image', 'question', 'answer', 'prediction', 'n_hop', 'has_scene_graph', 'split'])
    return csv_file, csv_writer


def run_pipeline(vqatask, path_to_dataset, output_dir_name, limit=0, start_at=0, splits=None):
    csv_file, csv_writer = init_csv_file(output_dir_name)

    if splits is None:
        splits = ['train', 'test']

    for split in splits:
        print('==========', split)
        json_data = stream_data(f'{path_to_dataset}/{split}.json', limit=limit, start_at=start_at)

        i = 0
        for d in json_data:
            i += 1

            if i == 1 or i % 10 == 0:
                print(f"[{i}]: {d['image_id']}")

            # split into smaller CSV file every 1000 records
            if i % 1000 == 0:
                csv_file.close()
                csv_file, csv_writer = init_csv_file(output_dir_name)

            local_img_path = f"{split}/{d['image_id']}.jpg"
            img_path = f"{path_to_dataset}/" + local_img_path

            prediction = vqatask(img_path, d['question'])
            answers = d['answers']
            csv_writer.writerow([d['image_id'], local_img_path, d['question'], answers, prediction, d['n_hop'], d['has_scene_graph'], split])

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


def analysis_result(csv_file):
    total = 0
    correct = 0
    total_by_hop = {}
    correct_by_hop = {}
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1

            n_hop = row['n_hop']
            if n_hop not in total_by_hop:
                total_by_hop[n_hop] = 0
                correct_by_hop[n_hop] = 0
            total_by_hop[n_hop] += 1

            answer_str = row['answer'].lower()
            prediction_str = row['prediction'].lower()
            answer = ast.literal_eval(answer_str)
            prediction = ast.literal_eval(prediction_str)

            is_correct = prediction[0] in answer
            if is_correct:
                correct += 1
                correct_by_hop[n_hop] += 1

    print('Total:', total, ', Correct:', correct, ' Accuracy:', f'{correct/total:.2f}')
    for h in correct_by_hop.keys():
        print(f'{h}-hop:', f'{correct_by_hop[h]/total_by_hop[n_hop]:.2f}')


if __name__ == '__main__':
    for csvfile in Path('result').iterdir():
        print(csvfile.name)
        analysis_result(f'result/{csvfile.name}')
        print('=====')


