import ast
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

METRICS = ['exact_match', 'substring', 'similarity']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)


class Score:
    exact_match: int = 0
    substring: int = 0
    similarity: float = 0.0
    vqa_acc: float = 0.0

    def __getitem__(self, item):
        if item == 'exact_match':
            return self.exact_match
        if item == 'substring':
            return self.substring
        if item == 'similarity':
            return self.similarity
        if item == 'vqa_acc':
            return self.vqa_acc

    def __setitem__(self, key, value):
        if key == 'exact_match':
            self.exact_match = value
        if key == 'substring':
            self.substring = value
        if key == 'similarity':
            self.similarity = value
        if key == 'vqa_acc':
            self.vqa_acc = value

    def __str__(self):
        return f'exact_match: {self.exact_match}; substring: {self.substring}; similarity: {self.similarity:.2f};'


# gt is the ground truth list of answers
def exact_match_score(pred, gt):
    return 1 if pred in gt else 0


def substring_score(pred, gt):
    for s in gt:
        gt_words = s.split()
        if pred in gt_words:
            return 1
    return 0


# https://huggingface.co/tasks/sentence-similarity
def similarity_score(pred, gt):
    max_score = 0
    pred_emb = similarity_model.encode(pred, convert_to_tensor=True, device=device)
    for s in gt:
        emb = similarity_model.encode(s, convert_to_tensor=True, device=device)
        current_score = util.pytorch_cos_sim(pred_emb, emb).item()
        if current_score > max_score:
            max_score = current_score
    return max_score


# https://visualqa.org/evaluation.html
def vqa_acc(pred, gt):
    return 0.0


# depending on the model, answer might be given within a complete sentence. e.g.: [answer] The length is 300 meters
# we need to extract "The length is 300 meters" only
def extract_answer(answer_text):
    if answer_text.startswith('Answer:'):
        return answer_text[7:].strip()
    return answer_text.strip()


def get_ratio(a, b):
    if b == 0:
        return 0
    return a / b


def analysis_result(list_of_result_dir, limit=0):
    total = 0
    score = Score()

    total_by_hop = {}
    score_by_hop = {}

    total_by_scene_graph = {
        'with': 0,
        'without': 0
    }
    score_by_scene_graph = {
        'with': Score(),
        'without': Score()
    }

    total_by_ds = {
        'VG': 0,
        'GLDv2': 0
    }
    score_by_ds = {
        'VG': Score(),
        'GLDv2': Score()
    }

    count = 0
    for folder in list_of_result_dir:
        if 0 < limit < count:
            break
        for csvfile in Path(folder).iterdir():
            if 0 < limit < count:
                break
            csv_file = f'{csvfile.parent}/{csvfile.name}'
            with open(csv_file) as f:
                count += 1
                if count % 100 == 0:
                    print(count)

                reader = csv.DictReader(f)
                for row in reader:
                    # there's a bug that the answer set is empty, ignore them
                    answer_str = row['answer'].lower()
                    answer = ast.literal_eval(answer_str)
                    if len(answer) == 0:
                        continue

                    total += 1

                    n_hop = row['n_hop']
                    if n_hop not in total_by_hop:
                        total_by_hop[n_hop] = 0
                        score_by_hop[n_hop] = Score()
                    total_by_hop[n_hop] += 1

                    if row['has_scene_graph']:
                        total_by_scene_graph['with'] += 1
                    else:
                        total_by_scene_graph['without'] += 1

                    ds_name = 'VG' if row['id'].startswith('VG_') else 'GLDv2'
                    total_by_ds[ds_name] += 1

                    prediction_str = row['prediction'].lower()
                    if prediction_str.startswith('['):
                        prediction_str = ast.literal_eval(prediction_str)[0]
                    prediction = extract_answer(prediction_str)

                    # compute all scores
                    current_score = Score()
                    if 'exact_match' in METRICS:
                        current_score.exact_match = exact_match_score(prediction, answer)
                    if 'substring' in METRICS:
                        current_score.substring = substring_score(prediction, answer)
                    if 'similarity' in METRICS:
                        current_score.similarity = similarity_score(prediction, answer)

                    for s in METRICS:
                        score[s] += current_score[s]
                        score_by_hop[n_hop][s] += current_score[s]
                        score_by_scene_graph['with' if row['has_scene_graph'] else 'without'][s] += current_score[s]
                        score_by_ds[ds_name][s] += current_score[s]

    print('Total:', total, ', Score:', score)
    for s in METRICS:
        print('=====', s)
        print('Acc:', f'{get_ratio(score[s], total):.2f}')
        for h in score_by_hop.keys():
            print(f'{h}-hop:', f'{get_ratio(score_by_hop[h][s], total_by_hop[n_hop]):.2f}')
        print('W/ Scene graph:', f"{get_ratio(score_by_scene_graph['with'][s], total_by_scene_graph['with']):.2f}")
        print('W/O Scene graph:',
              f"{get_ratio(score_by_scene_graph['without'][s], total_by_scene_graph['without']):.2f}")
        for ds in total_by_ds.keys():
            print(ds, f"{get_ratio(score_by_ds[ds][s], total_by_ds[ds]):.2f}")


if __name__ == '__main__':
    analysis_result(['result/output_balanced_10', 'result/output_balanced_10_test'])

    # pred = 'America'
    # gt = ['United States']
    # scr = substring_score(pred, gt)
    # print(scr)
