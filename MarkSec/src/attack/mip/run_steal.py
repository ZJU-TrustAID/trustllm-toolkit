import json
import random
import os
from tqdm import tqdm
from src.attack.mip.greenlist_inversion_plus import GreenlistInversorPlus
from src.util.env import get_project_root
import pyomo.environ as pyomo


class MyAttacker(GreenlistInversorPlus):
    def __init__(self, model_path, solver_type='gurobi'):
        super().__init__(tokenizer_tag=model_path)

        # self.vocab_size = self.tokenizer.vocab_size
        self.vocab_size = 50272  # for OPT models
        self.token_color_list = []
        self.token_index = list(range(self.vocab_size))
        self.token_dict = {id: 1 for id in range(self.vocab_size)}

        self.log = None
        self.set_solver(TimeLimit=3600, solver_type=solver_type)

    def log_info(self, info):
        print(f'[MIP Log] {str(info)}')

    def load_my_dataset(self, jsonl_path, max_samples=2000):
        print(f'[MIP] Loading data from {jsonl_path}...')
        wm_data, nl_data = [], []

        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        if len(lines) > max_samples:
            random.seed(42)
            lines = random.sample(lines, max_samples)

        for line in tqdm(lines, desc='Tokenizing'):
            try:
                item = json.loads(line)
                wm_text = item.get('generated_only', item.get('watermarked_text', ''))
                if wm_text:
                    ids = self.tokenizer.encode(wm_text, add_special_tokens=False)
                    if len(ids) > 10:
                        wm_data.append(ids)

                nl_text = item.get('clean_text', '').replace(item.get('prompt', ''), '')
                if nl_text:
                    ids = self.tokenizer.encode(nl_text, add_special_tokens=False)
                    if len(ids) > 10:
                        nl_data.append(ids)
            except Exception:
                continue

        min_len = min(len(wm_data), len(nl_data))
        self.dataset = wm_data[:min_len] + nl_data[:min_len]
        self.wm_size = min_len
        self.nl_size = min_len
        self.dataset_size = len(self.dataset)

        self.wm_sentence_index = list(range(self.wm_size))
        self.nl_sentence_index = list(range(self.wm_size, self.dataset_size))
        self.sentence_len_list = [len(sen) for sen in self.dataset]

        print(f'[MIP] Data Ready: {self.wm_size} WM + {self.nl_size} NL')

    def save_solution(self, dir_path, file_name):
        # 1. Extract stolen result
        token_color = {}
        key_id = 0  # Single key
        token_color[str(key_id)] = {}

        stolen_green_indices = set()

        print('[MIP] Extracting solution...')
        for token_id in self.token_index:
            try:
                val = pyomo.value(self.model.key_token_color[key_id, token_id])
                is_green = 1 if val >= 0.5 else 0
            except Exception:
                is_green = 0

            token_color[str(key_id)][str(token_id)] = ['', is_green]
            if is_green == 1:
                stolen_green_indices.add(token_id)

        # 2. Save file
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'w') as f:
            json.dump(token_color, f, indent=4)
        print(f'✅ Stolen greenlist saved to: {file_path}')

        # 3. Immediate evaluation
        if self.true_green:
            print('\n>>> Calculating Precision/Recall against Ground Truth...')
            # Get ground-truth greenlist (assume key `0`).
            # Compatible with historical export formats (list or dict).
            gt_data = self.true_green.get('0', self.true_green)

            true_green_indices = set()
            if isinstance(gt_data, list):
                true_green_indices = set(gt_data)
            else:
                # Dict format
                for tid in gt_data:
                    val = gt_data[tid]
                    flag = val[1] if isinstance(val, list) else val
                    if flag == 1:
                        true_green_indices.add(int(tid))

            # Compute metrics
            tp = len(stolen_green_indices.intersection(true_green_indices))
            stolen_size = len(stolen_green_indices)
            true_size = len(true_green_indices)

            precision = tp / stolen_size if stolen_size > 0 else 0
            recall = tp / true_size if true_size > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print('=' * 40)
            print(f'📊 Evaluation Result')
            print('=' * 40)
            print(f'Stolen Size: {stolen_size}')
            print(f'True Size:   {true_size}')
            print(f'Correct:     {tp}')
            print('-' * 40)
            print(f'🚀 Precision: {precision:.2%}')
            print(f'📈 Recall:    {recall:.2%}')
            print(f'⚖️  F1 Score:  {f1:.2%}')
            print('=' * 40)


def run():
    project_root = get_project_root()
    model_path = 'facebook/opt-6.7b'
    data_path = str(project_root / 'output' / 'corpus' / 'Unigram_opt-6.7b_vllm_g0.5_d2.0.jsonl')
    # Ground-truth red/green table path
    gt_path = str(
        project_root / 'output' / 'models' / 'ground_truth' / 'greenlist_opt-6.7b_0_0.5.json'
    )

    out_dir = str(project_root / 'output' / 'models' / 'mip_result')
    os.makedirs(out_dir, exist_ok=True)

    print('>>> Initializing Attacker...')
    attacker = MyAttacker(model_path, solver_type='gurobi')

    # 1. Load data
    attacker.load_my_dataset(data_path, max_samples=2000)

    # 2. Load ground truth (critical step)
    print(f'>>> Loading Ground Truth from {gt_path}...')
    attacker.load_true_green(gt_path)

    print('>>> Building MIP Model...')
    attacker.gamma = 0.5
    attacker.z_threshold = 4.0
    attacker.set_model(
        sentence_up_num=1.0,
        sentence_down_num=0.9,
        gamma_flag=False,
        oracle_flag=False,
        expect_green_size=0.5,
        key_list=[0],
    )

    print('>>> Stage 1: Estimating Bounds...')
    attacker.solve_wm_nl_sum_green(TimeLimit=3600, MIPGap=0.05)

    print('>>> Stage 2: Extracting Green List...')
    attacker.solve_green_num(max_min=False, TimeLimit=3600, MIPGap=0.05)

    print('>>> Saving Results...')
    attacker.save_solution(out_dir, 'stolen_greenlist.json')


if __name__ == '__main__':
    run()
