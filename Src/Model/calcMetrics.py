import json
from transformers import AutoTokenizer
from Levenshtein import distance
from nltk.translate.bleu_score import sentence_bleu
import difflib

def add_element_to_interval(element_id, element, intervals, interval_elements):
    interval_index = element_id // 100
    if interval_index >= len(intervals):
        print(element_id)
        print("out of range.")
        return

    interval = intervals[interval_index]
    interval_elements[interval].append(element)

def calculate_similarity(str1, str2):
    lev_distance = distance(str1, str2)
    similarity = 1 - lev_distance / max(len(str1), len(str2))
    return similarity

def get_diff(o_file, m_file):
    o_lines = o_file.split("\n")
    m_lines = m_file.split("\n")
    differ = difflib.Differ()
    diff_lines = list(differ.compare(o_lines, m_lines))
    
    a = []
    b = []
    for line in diff_lines:
        if line.startswith('+ '):
            a.append(line[2:])
        if line.startswith('- '):
            b.append(line[2:])
    return "\n".join(a), "\n".join(b)

def calc_bleu_diff(a, b):
    a, b = get_diff(a, b)
    if a == '' and b == '':
        return 1
    else:
        return sentence_bleu([a.split(" ")], b.split(" "))

ReLOpt_cl_path = "ModelRes/ReLOpt_CL.json"
ReLOpt_cg_path = "ModelRes/ReLOpt_CG.json"
base_model_path = "ModelRes/BaseModels.json"
with open(ReLOpt_cl_path) as f:
    ReLOpt_cl = json.loads(f.read())
with open(ReLOpt_cg_path) as f:
    ReLOpt_cg = json.loads(f.read())
with open(base_model_path) as f:
    base_model = json.loads(f.read())

unopt_ir_2_truth_base_ReLOpt = {}
for unoptIR in ReLOpt_cl:
    assert unoptIR in base_model and unoptIR in ReLOpt_cg
    unopt_ir_2_truth_base_ReLOpt[unoptIR] = {"ReLOpt_cl_truth": ReLOpt_cl[unoptIR]["ReLOpt_truth"], "ReLOpt_cl": ReLOpt_cl[unoptIR]["ReLOpt"]}
    unopt_ir_2_truth_base_ReLOpt[unoptIR] = {"ReLOpt_cg_truth": ReLOpt_cg[unoptIR]["ReLOpt_truth"], "ReLOpt_cg": ReLOpt_cg[unoptIR]["ReLOpt"]}
    unopt_ir_2_truth_base_ReLOpt[unoptIR]["base_truth"] = base_model[unoptIR]["base_truth"]
    unopt_ir_2_truth_base_ReLOpt[unoptIR]["codellama"] = base_model[unoptIR]["codellama"]
    unopt_ir_2_truth_base_ReLOpt[unoptIR]["codegemma"] = base_model[unoptIR]["codegemma"]

# check
for unopt_ir in unopt_ir_2_truth_base_ReLOpt:
    assert len(unopt_ir_2_truth_base_ReLOpt[unopt_ir]) == 5

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token_id = 2
tokenizer.padding_side = "left"

interval_metrics = {interval: None for interval in intervals}
# cnt = 0
for interval in interval_elements:
    ReLOpt_cl_bleu = 0
    ReLOpt_cl_bleu_diff = 0
    ReLOpt_cl_edit_sim = 0

    ReLOpt_cg_bleu = 0
    ReLOpt_cg_bleu_diff = 0
    ReLOpt_cg_edit_sim = 0

    codellama_bleu = 0
    codellama_bleu_diff = 0
    codellama_edit_sim = 0

    codegemma_bleu = 0
    codegemma_bleu_diff = 0
    codegemma_edit_sim = 0

    for e in interval_elements[interval]:
        ReLOpt_cl_truth = e['ReLOpt_cl_truth']
        ReLOpt_cl = e['ReLOpt_cl']
        ReLOpt_cg_truth = e['ReLOpt_cg_truth']
        ReLOpt_cg = e['ReLOpt_cg']
        base_truth = e['base_truth']
        codellama = e['codellama']
        codegemma = e['codegemma']
        
        ReLOpt_cl_bleu += sentence_bleu([ReLOpt_cl_truth.split(" ")], ReLOpt_cl.split(" "))
        ReLOpt_cl_bleu_diff += calc_bleu_diff(ReLOpt_cl_truth, ReLOpt_cl)
        ReLOpt_cl_edit_sim += calculate_similarity(ReLOpt_cl_truth, ReLOpt_cl)

        ReLOpt_cg_bleu += sentence_bleu([ReLOpt_cg_truth.split(" ")], ReLOpt_cg.split(" "))
        ReLOpt_cg_bleu_diff += calc_bleu_diff(ReLOpt_cg_truth, ReLOpt_cg)
        ReLOpt_cg_edit_sim += calculate_similarity(ReLOpt_cg_truth, ReLOpt_cg)

        codellama_bleu += sentence_bleu([base_truth.split(" ")], codellama.split(" "))
        codellama_bleu_diff += calc_bleu_diff(base_truth, codellama)
        codellama_edit_sim += calculate_similarity(base_truth, codellama)

        codegemma_bleu += sentence_bleu([base_truth.split(" ")], codegemma.split(" "))
        codegemma_bleu_diff += calc_bleu_diff(base_truth, codegemma)
        codegemma_edit_sim += calculate_similarity(base_truth, codegemma)
    total = len(interval_elements[interval])
    if total == 0:
        break
    # cnt += ReLOpt_em
    interval_metrics[interval] = {"ReLOpt_cl":{"bleu_diff": ReLOpt_cl_bleu_diff/total, "bleu": ReLOpt_cl_bleu/total, "editsim": ReLOpt_cl_edit_sim/total},
                                  "ReLOpt_cg":{"bleu_diff": ReLOpt_cg_bleu_diff/total, "bleu": ReLOpt_cg_bleu/total, "editsim": ReLOpt_cg_edit_sim/total},
                                  "codellama":{"bleu_diff": codellama_bleu_diff/total, "bleu": codellama_bleu/total, "editsim": codellama_edit_sim/total},
                                  "codegemma":{"bleu_diff": codegemma_bleu_diff/total, "bleu": codegemma_bleu/total, "editsim": codegemma_edit_sim/total}}

with open("metrics.json", "w") as f:
    json.dump(interval_metrics, f)