import pyomo.environ as pyomo
from src.attack.mip.util_mip import (
    load_json,
    save_json,
    to_string,
    print_info,
    count_z_score_bound,
    remove_repeat,
)
import numpy as np
import random
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import torch.nn.functional as F


class _SilentInfoLogger:
    def info(self, *_args, **_kwargs):
        return None


class _SilentLogger:
    def __init__(self):
        self.logger = _SilentInfoLogger()


class GreenlistInversorBase:
    def __init__(
        self,
        tokenizer_tag='',
        tokenizer=None,
        vocab_size=None,
        logger=None,
    ):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_tag, token=False)
        else:
            self.tokenizer = tokenizer

        resolved_vocab_size = vocab_size
        if resolved_vocab_size is None:
            resolved_vocab_size = getattr(self.tokenizer, 'vocab_size', None)
        if resolved_vocab_size is None and hasattr(self.tokenizer, '__len__'):
            try:
                resolved_vocab_size = len(self.tokenizer)
            except TypeError:
                resolved_vocab_size = None
        if resolved_vocab_size is None:
            raise ValueError('GreenlistInversorBase requires a tokenizer vocab size.')
        self.vocab_size = int(resolved_vocab_size)
        self.true_green = None
        self.device = 'cuda:0'
        self.token_color_list = []

        if logger is None:
            self.log = _SilentLogger()
        else:
            self.log = logger
        self.log_info('\n')

    def log_info(self, info):
        print_info(info)
        self.log.logger.info(info)

    def get_z_score_bound(self, len_sentence):
        return count_z_score_bound(len_sentence, self.z_threshold, self.gamma)

    def load_true_green(self, file_path):
        self.true_green = load_json(file_path)
        self.true_green_dict = {}
        for key in self.true_green:
            self.true_green_dict[key] = {}
            for idx in range(self.vocab_size):
                if idx in self.true_green[key]:
                    self.true_green_dict[key][idx] = 1
                else:
                    self.true_green_dict[key][idx] = 0

        return

    def count_rate(self, sentence_index_list):
        token_rate = {}
        for sen_id in sentence_index_list:
            for tmp_token in self.dataset[sen_id]:
                if tmp_token in token_rate:
                    token_rate[tmp_token] += 1
                else:
                    token_rate[tmp_token] = 1

        token_num = len(token_rate)
        for tmp_token in token_rate:
            token_rate[tmp_token] = token_rate[tmp_token] / token_num
        return token_rate

    def count_green_key(self, token_ids, key):
        green_num = 0
        for token in token_ids:
            if self.token_dict[token] == 0:
                continue
            green_num += self.true_green_dict[key][token]
        return green_num

    def count_green(self, token_ids):
        green_num = 0
        key = '1'
        for tmp_key in self.true_green:
            tmp_num = self.count_green_key(token_ids, tmp_key)

            if green_num < tmp_num:
                green_num = tmp_num
                key = tmp_key
        return green_num, key

    def count_green_vector(self, token_ids, key_id):
        tmp_vector = F.one_hot(torch.tensor(token_ids), self.vocab_size).to(self.device)
        return torch.sum(torch.mul(tmp_vector, self.true_green_vector[key_id]))

    def set_dataset(
        self,
        dir_path,
        file_name,
        input_type,
        wm_data_num=5000,
        nl_data_num=5000,
        gamma=0.25,
        z_threshold=4,
        key_num=1,
        wm_seed=123,
        nl_seed=123,
        rand_num=0,
        token_prop=(0, 1),
        rand_seed=123,
        true_key_list=[],
        query_flag=True,
        min_len=20,
    ):
        self.gamma = gamma
        self.z_threshold = z_threshold
        self.key_num = key_num
        self.true_key_list = true_key_list
        self.query_flag = query_flag
        self.log_info(to_string(('gamma', gamma)))
        self.log_info(to_string(('z_threshold', z_threshold)))
        self.log_info(to_string(('true_key_list', true_key_list)))
        self.log_info(to_string(('token_prop', token_prop)))
        self.log_info(to_string(('wm_data_num，nl_data_num', wm_data_num, nl_data_num)))
        self.log_info(to_string(('query_flag', query_flag)))

        input_file_path = os.path.join(dir_path, file_name + input_type)
        data_json = load_json(input_file_path)

        if isinstance(wm_data_num, int):
            wm_rand_num = 0
        else:
            wm_rand_num = wm_data_num - int(wm_data_num)
            wm_data_num = int(wm_data_num)
        if isinstance(nl_data_num, int):
            nl_rand_num = 0
        else:
            nl_rand_num = nl_data_num - int(nl_data_num)
            nl_data_num = int(nl_data_num)  # *(1+nl_rand_num))

        with_watermark_list = []
        without_watermark_list = []
        for data in data_json:
            if (
                ('redecoded_input' in data)
                and ('output_with_watermark' in data)
                and ('output_without_watermark' in data)
            ):
                with_watermark_list.append(data['output_with_watermark'])
                without_watermark_list.append(data['output_without_watermark'])

        tmp_wm_data = self.tokenizer.batch_encode_plus(with_watermark_list).data['input_ids']
        random.seed(wm_seed)
        random.shuffle(tmp_wm_data)
        tmp_wm_data = tmp_wm_data[0:wm_data_num]

        tmp_nl_data = self.tokenizer.batch_encode_plus(without_watermark_list).data['input_ids']
        random.seed(nl_seed)
        random.shuffle(tmp_nl_data)
        tmp_nl_data = tmp_nl_data[0:nl_data_num]

        token_rate = {idx: 0 for idx in range(self.vocab_size)}

        all_token_index = sorted(token_rate.items(), key=lambda x: x[1], reverse=True)
        all_token_index0 = all_token_index[0 : int(self.vocab_size * 0.5)]
        all_token_index1 = all_token_index[int(self.vocab_size * 0.5) :]
        token_start = int(token_prop[0] * self.vocab_size * 0.5)
        token_end = int(token_prop[1] * self.vocab_size * 0.5)
        random.seed(rand_seed)
        random.shuffle(all_token_index0)
        random.seed(rand_seed)
        random.shuffle(all_token_index1)
        tmp_token_index = (
            all_token_index0[token_start:token_end] + all_token_index1[token_start:token_end]
        )

        self.token_index = [t[0] for t in tmp_token_index]
        self.token_dict = {id: 0 for id in range(self.vocab_size)}
        for token_id in self.token_index:
            self.token_dict[token_id] = 1

        wm_data, nl_data = [], []
        for idx in tqdm(range(len(tmp_wm_data)), ncols=100, desc='load wm', leave=True):
            token_ids = tmp_wm_data[idx]
            # key=get_key(token_ids, key_num)
            token_ids = remove_repeat(token_ids)
            if len(token_ids) < min_len:
                continue

            if query_flag:
                green_num, key = self.count_green(token_ids)
                z_num = self.get_z_score_bound(self.sentence_len(token_ids))
                if green_num < z_num:
                    nl_data.append(token_ids)
                else:
                    wm_data.append(token_ids)
            else:
                wm_data.append(token_ids)

        for idx in tqdm(range(len(tmp_nl_data)), ncols=100, desc='load nl', leave=True):
            token_ids = tmp_nl_data[idx]
            token_ids = remove_repeat(token_ids)
            if len(token_ids) < min_len:
                continue

            if query_flag:
                green_num, key = self.count_green(token_ids)
                z_num = self.get_z_score_bound(self.sentence_len(token_ids))
                if green_num >= z_num:
                    wm_data.append(token_ids)
                else:
                    nl_data.append(token_ids)
            else:
                nl_data.append(token_ids)

        if rand_num > 0:
            rand_token = np.random.choice(self.vocab_size, size=rand_num, replace=False)
            rand_sentence = np.ones((rand_num, 50))
            rand_sentence = (rand_token * rand_sentence.T).T
            rand_sentence = rand_sentence.astype(np.int32)

            for idx in tqdm(range(rand_num), ncols=100, desc='load random', leave=True):
                rs = rand_sentence[idx]
                green_num, key = self.count_green(rs)
                z_num = self.get_z_score_bound(len(rs))
                if green_num >= z_num:
                    wm_data.append(rs.tolist())
                else:
                    nl_data.append(rs.tolist())

        if wm_rand_num > 0 and nl_rand_num > 0:
            perb_num = min(int(len(wm_data) * wm_rand_num), int(len(nl_data) * nl_rand_num))
            tmp_rand_wm = wm_data[-perb_num:]
            tmp_rand_nl = nl_data[-perb_num:]
            wm_data = wm_data[0:-perb_num] + tmp_rand_nl
            nl_data = nl_data[0:-perb_num] + tmp_rand_wm

        self.dataset = wm_data + nl_data
        self.dataset_size = len(self.dataset)
        self.wm_size = len(wm_data)
        self.nl_size = len(nl_data)

        self.wm_sentence_index = list(range(len(wm_data)))
        self.nl_sentence_index = list(range(len(wm_data), self.dataset_size, 1))
        self.sentence_len_list = [len(sen) for sen in self.dataset]

        self.label = np.hstack(
            (
                np.ones(len(wm_data)),
                np.zeros(len(nl_data)),
            )
        )

        print_info('log info: dataset loaded')

    def sentence_len(self, sentence):
        len_sen = sum([self.token_dict[token] for token in sentence])
        return len_sen

    def set_solver(self, TimeLimit=300, solver_type='gurobi', MIPGap=1e-4):
        self.solver = pyomo.SolverFactory(solver_type, solver_io='python')
        self.solver.options['TimeLimit'] = TimeLimit
        self.solver.options['LogToConsole'] = 1
        self.solver.options['OutputFlag'] = 1
        self.solver.options['Seed'] = 123
        self.solver.options['MIPGap'] = MIPGap

    def solve(self, TimeLimit=None, MIPGap=None, warmstart=True, MIPGapAbs=None):
        if TimeLimit is not None:
            self.solver.options['TimeLimit'] = TimeLimit
        if MIPGap is not None:
            self.solver.options['MIPGap'] = MIPGap
        if MIPGapAbs is not None:
            self.solver.options['MIPGapAbs'] = MIPGapAbs

        self.solver.solve(self.model, warmstart=warmstart)

    def save_solution(self, dir_path, file_name):
        green_num = {key_id: 0 for key_id in self.true_key_list}
        true_green_num = {key_id: 0 for key_id in self.true_key_list}
        if self.solver._solver_model.SolCount == 0:
            token_color = {
                key_id: {
                    token_id: (self.tokenizer.decode(token_id), 0) for token_id in self.token_index
                }
                for key_id in self.key_index
            }
        else:
            tmp_token_color = {}
            for idx in range(len(self.key_index)):
                key_id = self.key_index[idx]
                tmp_token_color[key_id] = {}
                for token_id in self.token_index:
                    try:
                        tmp_color = pyomo.value(self.model.key_token_color[key_id, token_id])
                    except Exception:
                        tmp_color = 0
                    if tmp_color >= 0.5:
                        tmp_color = 1
                    else:
                        tmp_color = 0

                    tmp_token_color[key_id][token_id] = (self.tokenizer.decode(token_id), tmp_color)
            token_color, green_num, true_green_num = self.match_token_color(tmp_token_color)

        self.log_info(to_string(('green_num', green_num)))

        for idx in range(len(self.true_key_list)):
            key_id = self.true_key_list[idx]
            if key_id not in green_num:
                continue
            if green_num[key_id] == 0:
                self.log_info(to_string(('true_green_num', key_id, true_green_num[key_id], 0)))
            else:
                green_acc = np.round(true_green_num[key_id] / green_num[key_id], 4)
                self.log_info(
                    to_string(('true_green_num', key_id, true_green_num[key_id], green_acc))
                )
        file_path = os.path.join(dir_path, file_name)
        for key_id in self.true_key_list:
            if key_id not in token_color:
                tmp_key = list(token_color.keys())[0]
                token_color[key_id] = token_color[tmp_key]
        self.token_color_list.append(token_color)
        save_json(token_color, file_path=file_path)

    def find_key4token_dict(self, tmp_token_color, token_color):
        same_num = [0] * len(self.true_key_list)
        green_num = 0
        for token in tmp_token_color:
            green_num += tmp_token_color[token][1]
            for idx in range(len(self.true_key_list)):
                key = self.true_key_list[idx]
                if key in token_color:
                    continue
                if tmp_token_color[token][1] == 1 and self.true_green_dict[str(key)][token] == 1:
                    same_num[idx] += 1
        return self.true_key_list[same_num.index(max(same_num))], max(same_num), green_num

    def match_token_color(self, tmp_token_color):
        green_num = {}
        for key1 in tmp_token_color:
            green_num[key1] = 0
            for token in tmp_token_color[key1]:
                if tmp_token_color[key1][token][1] == 1:
                    green_num[key1] += 1

        tmp_true_green_num = {}
        for key1 in tmp_token_color:
            for key2 in self.true_green_dict:
                tmp_true_green_num[(key1, key2)] = 0
                for token in tmp_token_color[key1]:
                    if (tmp_token_color[key1][token][1] == 1) and (
                        tmp_token_color[key1][token][1] == self.true_green_dict[key2][token]
                    ):
                        tmp_true_green_num[(key1, key2)] += 1

        token_color = {}
        true_green_num = {}
        for key2 in self.true_green_dict:
            tmp_score = 0
            tmp_key = -1
            for key1 in tmp_token_color:
                if key1 in token_color:
                    continue
                if tmp_true_green_num[(key1, key2)] >= tmp_score:
                    tmp_score = tmp_true_green_num[(key1, key2)]
                    tmp_key = key1
            if tmp_key == -1:
                continue
            token_color[tmp_key] = tmp_token_color[tmp_key]
            true_green_num[tmp_key] = tmp_true_green_num[(tmp_key, key2)]
        return token_color, green_num, true_green_num

    def compare_token_color(self, dir_path, file_name):
        file_path = os.path.join(dir_path, file_name)
        load_json(file_path=file_path)
        self.z_score_bound_list = [
            self.get_z_score_bound(self.sentence_len(sen)) for sen in self.dataset
        ]
        if self.true_green is None:
            self.green_num_list = [0 for _ in self.dataset]
        else:
            self.green_num_list = [self.count_green(sen)[0] for sen in self.dataset]


class GreenlistInversorPlus(GreenlistInversorBase):
    def set_model(
        self,
        sentence_up_num=1,
        sentence_down_num=1,
        perb_rate=0,
        expect_green_size=1,
        key_list=[],
        token_len=1,
        gamma_flag=True,
        oracle_flag=False,
        reals_flag=False,
        token_color_ini=0,
        min_max_green_bound=-1,
        min_green_num=2000,
        fixed_wm_sentence_key=None,
        use_token_weights=True,
    ):
        self.log_info(
            to_string(('sentence_up_num, sentence_down_num', sentence_up_num, sentence_down_num))
        )
        self.log_info(to_string(('expect_green_size', expect_green_size)))
        self.log_info(to_string(('key_list', key_list)))
        self.log_info(to_string(('token_len', token_len)))
        self.log_info(to_string(('gamma_flag, oracle_flag', gamma_flag, oracle_flag)))
        self.log_info(to_string(('reals_flag', reals_flag)))

        self.key_index = key_list
        self.gamma_flag = gamma_flag

        self.wm_token_rate = self.count_rate(self.wm_sentence_index)
        self.nl_token_rate = self.count_rate(self.nl_sentence_index)

        if isinstance(token_color_ini, dict):
            tmp_token_color_ini = {
                (key_id, token_id): token_color_ini[str(key_id)][token_id]
                for key_id in self.key_index
                for token_id in self.token_index
            }
            token_color_ini = tmp_token_color_ini
            self.log_info(to_string(('ground truth', 'True')))
        else:
            random.seed(123)
            randm_green = list(range(self.vocab_size))
            random.shuffle(randm_green)
            step = 0.2
            step1 = 0.3
            tmp_token_color_ini = {}
            for idx, key_id in enumerate(self.key_index):
                for token_id in self.token_index:
                    tmp_start = int(idx * step * self.vocab_size)
                    tmp_end = int(tmp_start + step1 * self.vocab_size)
                    if tmp_start < self.vocab_size and tmp_end > self.vocab_size:
                        tmp_list = (
                            randm_green[tmp_start:] + randm_green[0 : tmp_end % self.vocab_size]
                        )
                    elif tmp_start > self.vocab_size and tmp_end > self.vocab_size:
                        tmp_end = tmp_end % self.vocab_size
                        tmp_start = tmp_start % self.vocab_size
                        tmp_list = randm_green[tmp_start:tmp_end]
                    else:
                        tmp_list = randm_green[tmp_start:tmp_end]
                    if token_id in tmp_list:
                        tmp_token_color_ini[(key_id, token_id)] = 1
                    else:
                        tmp_token_color_ini[(key_id, token_id)] = 0
            token_color_ini = tmp_token_color_ini
            self.log_info(to_string(('ground truth', 'False')))

        self.z_score_bound_list = [
            self.get_z_score_bound(self.sentence_len(sen)) for sen in self.dataset
        ]
        # Prebuilt-corpus stealing does not require ground-truth greenlists.
        # Keep solver setup valid when true_green is not provided.
        if self.true_green is None:
            self.green_num_list = [0 for _ in self.dataset]
        else:
            self.green_num_list = [self.count_green(sen)[0] for sen in self.dataset]

        self.key_token_index = [
            (key_id, token_id) for key_id in self.key_index for token_id in self.token_index
        ]
        self.sentence_index = list(range(self.dataset_size))
        self.sentence_key_index = [
            (sentence_id, key_id)
            for sentence_id in self.sentence_index
            for key_id in self.key_index
        ]
        self.wm_sentence_key_index = [
            (sentence_id, key_id)
            for sentence_id in self.wm_sentence_index
            for key_id in self.key_index
        ]
        self.nl_sentence_key_index = [
            (sentence_id, key_id)
            for sentence_id in self.nl_sentence_index
            for key_id in self.key_index
        ]

        self.token_weight = {}
        for token_id in self.token_index:
            token = self.tokenizer.decode(token_id)
            token = token.replace(' ', '')
            if token in [
                'that',
                'the',
                'to',
                'of',
                'is',
                'are',
                'be',
                'on',
                'in',
                'it',
                'an',
                'and',
                'for',
                '</s>',
            ]:
                self.token_weight[token_id] = 2
                continue
            if len(token) <= 1:
                self.token_weight[token_id] = 5
                continue
            if len(token) <= 2:
                self.token_weight[token_id] = 4
                continue

            self.token_weight[token_id] = 1

            if (
                (token_id in self.nl_token_rate)
                and (token_id in self.wm_token_rate)
                and perb_rate == 0
                and len(key_list) == 1
            ):  #
                rate = self.nl_token_rate[token_id] / self.wm_token_rate[token_id]
                self.token_weight[token_id] = rate

        self.model = pyomo.ConcreteModel()

        if reals_flag:
            self.model.key_token_color = pyomo.Var(
                self.key_token_index, domain=pyomo.Reals, bounds=(0, 1), initialize=token_color_ini
            )
        else:
            self.model.key_token_color = pyomo.Var(
                self.key_token_index, domain=pyomo.Boolean, initialize=token_color_ini
            )
        self.model.sentence_key = pyomo.Var(self.sentence_key_index, domain=pyomo.Boolean)
        self.model.whether_sentence = pyomo.Var(self.sentence_index, domain=pyomo.Boolean)
        self.model.whether_key = pyomo.Var(self.key_index, domain=pyomo.Boolean)
        self.model.sen_green_bound = pyomo.Var(self.sentence_index, domain=pyomo.NonNegativeReals)

        def sum_whether_key_expr(model):
            return sum([model.whether_key[key_id] for key_id in self.key_index])

        self.model.sum_whether_key = pyomo.Expression(rule=sum_whether_key_expr)

        def sum_sentence_key_expr(model, sentence_id):
            return sum([model.sentence_key[sentence_id, key_id] for key_id in self.key_index])

        self.model.sum_sentence_key = pyomo.Expression(
            self.sentence_index, rule=sum_sentence_key_expr
        )

        def sum_sentence_key_con_expr(model, sentence_id):
            return model.sum_sentence_key[sentence_id] == 1

        self.model.sum_sentence_key_con = pyomo.Constraint(
            self.sentence_index, rule=sum_sentence_key_con_expr
        )

        if fixed_wm_sentence_key:
            for sen_id, key_id in self.wm_sentence_key_index:
                fixed_value = int(fixed_wm_sentence_key.get((sen_id, key_id), 0))
                self.model.sentence_key[(sen_id, key_id)].fix(fixed_value)

        def sum_green_key_expr(model, sentence_id, key_id):
            return sum(
                [
                    model.key_token_color[key_id, token_id]
                    for token_id in self.dataset[sentence_id]
                    if self.token_dict[token_id] == 1
                ]
            )

        self.model.sum_green_key = pyomo.Expression(
            self.sentence_key_index, rule=sum_green_key_expr
        )

        def whether_sum_green_wm_expr(model, sentence_id, key_id):
            if oracle_flag:
                return model.sum_green_key[sentence_id, key_id] >= (
                    model.sen_green_bound[sentence_id]
                    + (model.sentence_key[sentence_id, key_id] - 1)
                    * self.sentence_len_list[sentence_id]
                    * 2
                )
            return model.sum_green_key[sentence_id, key_id] >= (
                model.sen_green_bound[sentence_id]
                + (
                    model.sentence_key[sentence_id, key_id]
                    - 1
                    + model.whether_sentence[sentence_id]
                    - 1
                )
                * self.sentence_len_list[sentence_id]
                * 2
            )

        self.model.whether_sum_green_key_wm = pyomo.Constraint(
            self.wm_sentence_key_index, rule=whether_sum_green_wm_expr
        )

        if len(key_list) > 1:
            key_group = [
                (self.key_index[k1], self.key_index[k2 % len(self.key_index)])
                for k1 in range(len(self.key_index))
                for k2 in range(k1 + 1, k1 + len(self.key_index))
            ]

            def sum_token_key_expr(model, key_id):
                return sum(
                    [model.key_token_color[key_id, token_id] for token_id in self.token_index]
                )

            self.model.sum_token_key = pyomo.Expression(self.key_index, rule=sum_token_key_expr)

            def sum_token_key_con_expr(model, key_id1, key_id2):
                return (model.sum_token_key[key_id1] - model.sum_token_key[key_id2]) <= 100

            self.model.sum_token_key_con = pyomo.Constraint(key_group, rule=sum_token_key_con_expr)

            def sum_token_key_con2_expr(model, key_id):
                return model.sum_token_key[key_id] >= min_green_num

            self.model.sum_token_key_con2 = pyomo.Constraint(
                self.key_index, rule=sum_token_key_con2_expr
            )

        if self.gamma_flag:

            def sen_green_bound_wm_con_expr(model, sentence_id):
                if oracle_flag:
                    return model.sen_green_bound[sentence_id] == self.green_num_list[sentence_id]
                return (
                    model.sen_green_bound[sentence_id]
                    >= self.z_score_bound_list[sentence_id] * model.whether_sentence[sentence_id]
                )

            self.model.sen_green_bound_wm_con = pyomo.Constraint(
                self.wm_sentence_index, rule=sen_green_bound_wm_con_expr
            )

        def sum_sen_green_bound_wm_expr(model):
            return sum(
                [model.sen_green_bound[sentence_id] for sentence_id in self.wm_sentence_index]
            )

        self.model.sum_sen_green_bound_wm = pyomo.Expression(rule=sum_sen_green_bound_wm_expr)

        def whether_sum_green_nl_expr(model, sentence_id, key_id):
            if oracle_flag:
                return (
                    model.sum_green_key[sentence_id, key_id] <= model.sen_green_bound[sentence_id]
                )
            return model.sum_green_key[sentence_id, key_id] <= (
                model.sen_green_bound[sentence_id]
                + (1 - model.whether_sentence[sentence_id])
                * self.sentence_len_list[sentence_id]
                * 2
            )

        self.model.whether_sum_green_nl = pyomo.Constraint(
            self.nl_sentence_key_index, rule=whether_sum_green_nl_expr
        )

        if self.gamma_flag:

            def sen_green_bound_nl_con_expr(model, sentence_id):
                if oracle_flag:
                    return model.sen_green_bound[sentence_id] == self.green_num_list[sentence_id]
                return model.sen_green_bound[sentence_id] <= (
                    self.z_score_bound_list[sentence_id]
                    + (1 - model.whether_sentence[sentence_id])
                    * self.sentence_len_list[sentence_id]
                    * 2
                )

            self.model.sen_green_bound_nl_con = pyomo.Constraint(
                self.nl_sentence_index, rule=sen_green_bound_nl_con_expr
            )
        if gamma_flag:
            self.model.abs_sen_green_bound = pyomo.Var(
                self.nl_sentence_index, domain=pyomo.NonNegativeReals
            )

            def abs_sen_green_bound_con1_expr(model, sentence_id):
                return model.abs_sen_green_bound[sentence_id] >= (
                    model.sen_green_bound[sentence_id]
                    - self.sentence_len_list[sentence_id] * self.gamma
                )

            self.model.abs_sen_green_bound_con1 = pyomo.Constraint(
                self.nl_sentence_index, rule=abs_sen_green_bound_con1_expr
            )

            def abs_sen_green_bound_con2_expr(model, sentence_id):
                return model.abs_sen_green_bound[sentence_id] >= (
                    -model.sen_green_bound[sentence_id]
                    + self.sentence_len_list[sentence_id] * self.gamma
                )

            self.model.abs_sen_green_bound_con2 = pyomo.Constraint(
                self.nl_sentence_index, rule=abs_sen_green_bound_con2_expr
            )

        def sum_sen_green_bound_nl_expr(model):
            return sum(
                [model.sen_green_bound[sentence_id] for sentence_id in self.nl_sentence_index]
            )

        self.model.sum_sen_green_bound_nl = pyomo.Expression(rule=sum_sen_green_bound_nl_expr)

        if gamma_flag:

            def sum_abs_sen_green_bound_nl_expr(model):
                return sum(
                    [
                        model.abs_sen_green_bound[sentence_id]
                        for sentence_id in self.nl_sentence_index
                    ]
                )

            self.model.sum_abs_sen_green_bound_nl = pyomo.Expression(
                rule=sum_abs_sen_green_bound_nl_expr
            )

        def whether_bound_con_expr(model, sentence_id):
            return (
                model.whether_sentence[sentence_id] * self.sentence_len_list[sentence_id]
                >= model.sen_green_bound[sentence_id]
            )

        self.model.whether_bound_con = pyomo.Constraint(
            self.sentence_index, rule=whether_bound_con_expr
        )

        if not oracle_flag and perb_rate > 0:
            self.model.min_wm_sen_green_bound = pyomo.Var(domain=pyomo.NonNegativeReals)
            self.model.max_nl_sen_green_bound = pyomo.Var(domain=pyomo.NonNegativeReals)

            def min_wm_sen_green_bound_con_expr(model, sentence_id):
                return model.min_wm_sen_green_bound <= (
                    model.sen_green_bound[sentence_id] / self.sentence_len_list[sentence_id]
                    + 1
                    - model.whether_sentence[sentence_id]
                )

            self.model.min_wm_sen_green_bound_con = pyomo.Constraint(
                self.wm_sentence_index, rule=min_wm_sen_green_bound_con_expr
            )

            def max_nl_sen_green_bound_con_expr(model, sentence_id):
                return model.max_nl_sen_green_bound >= (
                    model.sen_green_bound[sentence_id] / self.sentence_len_list[sentence_id]
                    + model.whether_sentence[sentence_id]
                    - 1
                )

            self.model.max_nl_sen_green_bound_con = pyomo.Constraint(
                self.nl_sentence_index, rule=max_nl_sen_green_bound_con_expr
            )

            if perb_rate > 0:
                self.model.min_max_sen_green_bound_con2 = pyomo.Constraint(
                    expr=(self.model.min_wm_sen_green_bound - self.model.max_nl_sen_green_bound)
                    >= min_max_green_bound
                )

        def sum_whether_sentence_expr(model):
            return sum(
                [model.whether_sentence[sentence_id] for sentence_id in self.nl_sentence_index]
            ) + sum([model.whether_sentence[sentence_id] for sentence_id in self.wm_sentence_index])

        self.model.sum_whether_sentence = pyomo.Expression(rule=sum_whether_sentence_expr)

        def sum_whether_wm_sentence_expr(model):
            return sum(
                [model.whether_sentence[sentence_id] for sentence_id in self.wm_sentence_index]
            )

        self.model.sum_whether_wm_sentence = pyomo.Expression(rule=sum_whether_wm_sentence_expr)

        def sum_whether_wm_sentence_up_con_expr(model):
            return model.sum_whether_wm_sentence <= sentence_up_num * self.wm_size

        self.model.sum_whether_wm_sentence_up_con = pyomo.Constraint(
            rule=sum_whether_wm_sentence_up_con_expr
        )

        def sum_whether_wm_sentence_down_con_expr(model):
            return model.sum_whether_wm_sentence >= sentence_down_num * self.wm_size

        self.model.sum_whether_wm_sentence_down_con = pyomo.Constraint(
            rule=sum_whether_wm_sentence_down_con_expr
        )

        def sum_whether_nl_sentence_expr(model):
            return sum(
                [model.whether_sentence[sentence_id] for sentence_id in self.nl_sentence_index]
            )

        self.model.sum_whether_nl_sentence = pyomo.Expression(rule=sum_whether_nl_sentence_expr)

        def sum_whether_nl_sentence_up_con_expr(model):
            if perb_rate > 0:
                return model.sum_whether_nl_sentence <= sentence_up_num * self.nl_size
            else:
                return (
                    model.sum_whether_nl_sentence
                    <= max((sentence_up_num + perb_rate), 0.9) * self.nl_size
                )

        self.model.sum_whether_nl_sentence_up_con = pyomo.Constraint(
            rule=sum_whether_nl_sentence_up_con_expr
        )

        def sum_whether_nl_sentence_down_con_expr(model):
            if perb_rate > 0:
                return model.sum_whether_nl_sentence >= sentence_down_num * self.nl_size
            else:
                return (
                    model.sum_whether_nl_sentence
                    >= max((sentence_down_num + perb_rate), 0.8) * self.nl_size
                )

        self.model.sum_whether_nl_sentence_down_con = pyomo.Constraint(
            rule=sum_whether_nl_sentence_down_con_expr
        )

        def sum_green_token_expr(model):
            return sum(
                [
                    model.key_token_color[key_id, token_id]
                    * (self.token_weight[token_id] if use_token_weights else 1)
                    for key_id in self.key_index
                    for token_id in self.token_index
                    if self.token_dict[token_id] == 1
                ]
            )

        self.model.sum_green_token = pyomo.Expression(rule=sum_green_token_expr)

        def sum_green_token_each_expr(model, key_id):
            return sum(
                [
                    model.key_token_color[key_id, token_id]
                    for token_id in self.token_index
                    if self.token_dict[token_id] == 1
                ]
            )

        self.model.sum_green_token_each = pyomo.Expression(
            self.key_index, rule=sum_green_token_each_expr
        )

        def sum_green_token_each_con_expr(model, key_id):
            if gamma_flag:
                return (
                    model.sum_green_token_each[key_id]
                    <= self.vocab_size * self.gamma * expect_green_size
                )
            else:
                return model.sum_green_token_each[key_id] <= self.vocab_size * expect_green_size

        self.model.sum_green_token_each_con = pyomo.Constraint(
            self.key_index, rule=sum_green_token_each_con_expr
        )

        print_info('log info: model init')

    def solve_wm_nl_sum_green(
        self,
        TimeLimit=None,
        MIPGap=None,
        MIPGapAbs=None,
        wm_bound=0.95,
        nl_bound=1.05,
        lock_sentence=False,
        bound_flag=True,
    ):
        self.log_info(to_string(('TimeLimit, MIPGap: ', TimeLimit, MIPGap)))
        self.log_info(to_string(('wm_bound, nl_bound: ', wm_bound, nl_bound)))
        self.log_info(to_string(('lock_sentence: ', lock_sentence)))
        if self.gamma_flag:
            self.model.wm_nl_sum_green_obj = pyomo.Objective(
                expr=(
                    self.model.sum_sen_green_bound_wm / self.wm_size
                    - self.model.sum_abs_sen_green_bound_nl / self.nl_size
                ),
                sense=pyomo.maximize,
            )
        else:
            self.model.wm_nl_sum_green_obj = pyomo.Objective(
                expr=(
                    self.model.sum_sen_green_bound_wm / self.wm_size
                    - self.model.sum_sen_green_bound_nl / self.nl_size
                ),
                sense=pyomo.maximize,
            )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap, MIPGapAbs=MIPGapAbs)
        self.model.wm_nl_sum_green_obj.deactivate()

        try:
            sum_sen_green_bound_wm = pyomo.value(self.model.sum_sen_green_bound_wm)
            sum_sen_green_bound_nl = pyomo.value(self.model.sum_sen_green_bound_nl)
            sum_whether_wm_sentence = pyomo.value(self.model.sum_whether_wm_sentence)
            sum_whether_nl_sentence = pyomo.value(self.model.sum_whether_nl_sentence)
        except Exception:
            return False

        mean_g_o_i_wm = np.mean([self.green_num_list[idx] for idx in self.wm_sentence_index])
        mean_g_o_i_nl = np.mean([self.green_num_list[idx] for idx in self.nl_sentence_index])

        mean_g_i_wm = np.mean([self.z_score_bound_list[idx] for idx in self.wm_sentence_index])
        mean_g_i_nl = np.mean([self.z_score_bound_list[idx] for idx in self.nl_sentence_index])

        mean_bi_wm = np.mean(
            [
                pyomo.value(self.model.sen_green_bound[idx])
                for idx in self.wm_sentence_index
                if pyomo.value(self.model.sen_green_bound[idx]) > 0
            ]
        )
        mean_bi_nl = np.mean(
            [
                pyomo.value(self.model.sen_green_bound[idx])
                for idx in self.nl_sentence_index
                if pyomo.value(self.model.sen_green_bound[idx]) > 0
            ]
        )

        self.log_info(to_string(('mean_g_o_i_wm: ', np.round(mean_g_o_i_wm, 2))))
        self.log_info(to_string(('mean_g_o_i_nl: ', np.round(mean_g_o_i_nl, 2))))
        self.log_info(to_string(('mean_g_i_wm: ', np.round(mean_g_i_wm, 2))))
        self.log_info(to_string(('mean_g_i_nl: ', np.round(mean_g_i_nl, 2))))
        self.log_info(to_string(('mean_bi_wm: ', np.round(mean_bi_wm, 2))))
        self.log_info(to_string(('mean_bi_nl: ', np.round(mean_bi_nl, 2))))

        self.log_info(to_string(('sum_whether_nl_sentence: ', sum_whether_nl_sentence)))
        self.log_info(to_string(('sum_whether_wm_sentence: ', sum_whether_wm_sentence)))

        self.model.new_sum_sen_green_bound_wm_con = pyomo.Constraint(
            expr=self.model.sum_sen_green_bound_wm >= sum_sen_green_bound_wm * wm_bound
        )

        self.model.new_sum_sen_green_bound_nl_con = pyomo.Constraint(
            expr=self.model.sum_sen_green_bound_nl <= sum_sen_green_bound_nl * nl_bound
        )

        whether_sentence = None
        if lock_sentence or len(self.key_index) > 1:
            whether_sentence = {
                sentence_id: pyomo.value(self.model.whether_sentence[sentence_id])
                for sentence_id in self.sentence_index
            }

        if lock_sentence:
            self.model.sum_whether_wm_sentence_obj_con = pyomo.Constraint(
                expr=self.model.sum_whether_wm_sentence >= sum_whether_wm_sentence
            )
            self.model.sum_whether_nl_sentence_obj_con = pyomo.Constraint(
                expr=self.model.sum_whether_nl_sentence >= sum_whether_nl_sentence
            )
        if len(self.key_index) > 1 and whether_sentence is not None and self.true_green:
            sentence_key = {
                (sen_id, key_id): pyomo.value(self.model.sentence_key[(sen_id, key_id)])
                for (sen_id, key_id) in self.wm_sentence_key_index
            }

            true_key_rate = 0
            sen_num = 0
            for sen_id in self.wm_sentence_index:
                if whether_sentence[sen_id] == 0:
                    continue
                sen_num += 1
                _, tmp_true_key = self.count_green(self.dataset[sen_id])
                tmp_list = [sentence_key[(sen_id, key_id)] for key_id in self.key_index]
                tmp_idx = np.argmax(tmp_list)
                if list(self.true_green.keys())[tmp_idx] == tmp_true_key:
                    true_key_rate += 1
            if sen_num > 0:
                self.log_info(to_string(('true_key_rate: ', np.round(true_key_rate / sen_num, 4))))
        elif len(self.key_index) > 1:
            self.log_info('true_key_rate skipped: no ground-truth multi-key greenlist loaded')
        return True

    def solve_min_max_bound(self, TimeLimit=None, MIPGap=None):
        self.model.min_max_bound_obj = pyomo.Objective(
            expr=self.model.min_wm_sen_green_bound - self.model.max_nl_sen_green_bound,
            sense=pyomo.maximize,
        )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap)
        self.model.min_max_bound_obj.deactivate()

        min_wm_sen_green_bound = np.round(pyomo.value(self.model.min_wm_sen_green_bound), 2)
        max_nl_sen_green_bound = np.round(pyomo.value(self.model.max_nl_sen_green_bound), 2)

        self.log_info(to_string(('min_wm_sen_green_bound:', min_wm_sen_green_bound)))
        self.log_info(to_string(('max_nl_sen_green_bound:', max_nl_sen_green_bound)))

        self.model.min_max_bound_obj_con1 = pyomo.Constraint(
            expr=self.model.min_wm_sen_green_bound >= min_wm_sen_green_bound
        )
        self.model.min_max_bound_obj_con2 = pyomo.Constraint(
            expr=self.model.max_nl_sen_green_bound <= max_nl_sen_green_bound
        )

    def solve_sum_whether_sentence(self, TimeLimit=None, MIPGap=None, wm_bound=1, nl_bound=1):
        self.model.sum_whether_sentence_obj = pyomo.Objective(
            expr=self.model.sum_whether_sentence, sense=pyomo.maximize
        )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap, warmstart=False)
        self.model.sum_whether_sentence_obj.deactivate()

        sum_whether_nl_sentence = np.round(pyomo.value(self.model.sum_whether_nl_sentence), 2)
        sum_whether_wm_sentence = np.round(pyomo.value(self.model.sum_whether_wm_sentence), 2)
        self.log_info(to_string(('sum_whether_nl_sentence: ', sum_whether_nl_sentence)))
        self.log_info(to_string(('sum_whether_wm_sentence: ', sum_whether_wm_sentence)))

        sum_whether_sentence = pyomo.value(self.model.sum_whether_sentence)
        if sum_whether_sentence > 0:
            whether_sentence = {
                sentence_id: pyomo.value(self.model.whether_sentence[sentence_id])
                for sentence_id in self.sentence_index
            }

            def sum_whether_sentence_obj_con_expr(model, sentence_id):
                return self.model.whether_sentence[sentence_id] == whether_sentence[sentence_id]

            self.model.sum_whether_sentence_obj_con = pyomo.Constraint(
                self.sentence_index, rule=sum_whether_sentence_obj_con_expr
            )

    def solve_min_key_num(self, TimeLimit=None):
        self.model.sum_whether_key_obj = pyomo.Objective(
            expr=self.model.sum_whether_key, sense=pyomo.minimize
        )
        self.solve(TimeLimit=TimeLimit)
        sum_whether_key = pyomo.value(self.model.sum_whether_key)

        self.log_info(to_string(('least_key_num:', sum_whether_key)))

        self.model.sum_whether_key_obj.deactivate()
        self.model.sum_whether_key_obj_con = pyomo.Constraint(
            expr=self.model.sum_whether_key <= sum_whether_key
        )
        return

    def solve_green_num(
        self,
        max_min=False,
        TimeLimit=None,
        MIPGap=None,
        wm_bound=0.95,
        nl_bound=1.05,
        MIPGapAbs=None,
    ):
        if max_min:
            self.model.sum_green_token_obj = pyomo.Objective(
                expr=self.model.sum_green_token, sense=pyomo.maximize
            )
        else:
            self.model.sum_green_token_obj = pyomo.Objective(
                expr=self.model.sum_green_token, sense=pyomo.minimize
            )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap, MIPGapAbs=MIPGapAbs)
        self.model.sum_green_token_obj.deactivate()
        sum_green_token = pyomo.value(self.model.sum_green_token)

        if max_min:
            self.model.sum_green_token_obj_con = pyomo.Constraint(
                expr=self.model.sum_green_token >= sum_green_token * wm_bound
            )
        else:
            self.model.sum_green_token_obj_con = pyomo.Constraint(
                expr=self.model.sum_green_token <= sum_green_token * nl_bound
            )
        self.log_info(to_string(('sum_green_token:', sum_green_token)))
        return

    def solve_min_key_sentence_num(self, TimeLimit=None, MIPGap=None):
        self.model.min_key_sentence_num_obj = pyomo.Objective(
            expr=self.model.min_key_sentence_num, sense=pyomo.maximize
        )
        self.solve(TimeLimit=TimeLimit, MIPGap=MIPGap)
        self.model.min_key_sentence_num_obj.deactivate()
        min_key_sentence_num = pyomo.value(self.model.min_key_sentence_num)
        self.log_info(to_string(('min_key_sentence_num', min_key_sentence_num)))
        self.model.min_key_sentence_num_obj_con = pyomo.Constraint(
            expr=self.model.min_key_sentence_num >= min_key_sentence_num
        )
        return

    def count_token_green(self, sentence, token_color):
        key = 0
        green_num = []
        for tmp_key in token_color:
            tmp_green_num = 0
            for token_id in sentence:
                if token_id not in token_color[tmp_key]:
                    continue
                tmp_green_num += token_color[tmp_key][token_id][1]
            green_num.append(tmp_green_num)

        green_num = np.array(green_num) + 1e-4
        key = np.argmax(green_num)
        return green_num, self.key_index[key]

    def fix_sentence_key(self, random_flag=False, file_path=''):
        for sen_id, key_id in self.wm_sentence_key_index:
            self.model.sentence_key[(sen_id, key_id)].unfix()

        fix_wm_sentence_key = {
            (sentence_id, key_id): 0
            for sentence_id in self.wm_sentence_index
            for key_id in self.key_index
        }
        if random_flag:
            np.random.seed(123)
            for sen_id in self.wm_sentence_index:
                key_id = self.key_index[np.random.randint(len(self.key_index))]
                fix_wm_sentence_key[(sen_id, key_id)] = 1
        else:
            token_color = self.token_color_list[-1]
            for sen_id in self.wm_sentence_index:
                green_num, key_id = self.count_token_green(self.dataset[sen_id], token_color)
                fix_wm_sentence_key[(sen_id, key_id)] = 1

        for sen_id, key_id in self.wm_sentence_key_index:
            self.model.sentence_key[(sen_id, key_id)].fix(fix_wm_sentence_key[(sen_id, key_id)])
        self.log_info('fix done')
        if len(file_path) > 0:
            tmp_fix_wm_sentence_key = {
                str(sen_id) + '_' + str(key_id): fix_wm_sentence_key[(sen_id, key_id)]
                for (sen_id, key_id) in fix_wm_sentence_key
            }
            save_json(tmp_fix_wm_sentence_key, file_path)
        return

    def delete_comp(self):
        try:
            del self.model.wm_nl_sum_green_obj
        except Exception:
            pass
        try:
            del self.model.new_sum_sen_green_bound_wm_con
        except Exception:
            pass
        try:
            del self.model.new_sum_sen_green_bound_nl_con
        except Exception:
            pass

        try:
            del self.model.sum_whether_wm_sentence_obj_con
        except Exception:
            pass
        try:
            del self.model.sum_whether_nl_sentence_obj_con
        except Exception:
            pass
        try:
            del self.model.sum_green_token_obj
        except Exception:
            pass
        try:
            del self.model.sum_green_token_obj_con
        except Exception:
            pass
        return
