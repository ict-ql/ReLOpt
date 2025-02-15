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

coderefine_cl_path = "ModelRes/CodeRefine_CL.json"
coderefine_cg_path = "ModelRes/CodeRefine_CG.json"
base_model_path = "ModelRes/BaseModels.json"
with open(coderefine_cl_path) as f:
    coderefine_cl = json.loads(f.read())
with open(coderefine_cg_path) as f:
    coderefine_cg = json.loads(f.read())
with open(base_model_path) as f:
    base_model = json.loads(f.read())

unopt_ir_2_truth_base_coderefine = {}
for unoptIR in coderefine_cl:
    assert unoptIR in base_model and unoptIR in coderefine_cg
    unopt_ir_2_truth_base_coderefine[unoptIR] = {"coderefine_cl_truth": coderefine_cl[unoptIR]["coderefine_truth"], "coderefine_cl": coderefine_cl[unoptIR]["coderefine"]}
    unopt_ir_2_truth_base_coderefine[unoptIR] = {"coderefine_cg_truth": coderefine_cg[unoptIR]["coderefine_truth"], "coderefine_cg": coderefine_cg[unoptIR]["coderefine"]}
    unopt_ir_2_truth_base_coderefine[unoptIR]["base_truth"] = base_model[unoptIR]["base_truth"]
    unopt_ir_2_truth_base_coderefine[unoptIR]["codellama"] = base_model[unoptIR]["codellama"]
    unopt_ir_2_truth_base_coderefine[unoptIR]["codegemma"] = base_model[unoptIR]["codegemma"]

# check
for unopt_ir in unopt_ir_2_truth_base_coderefine:
    assert len(unopt_ir_2_truth_base_coderefine[unopt_ir]) == 5

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token_id = 2
tokenizer.padding_side = "left"

interval_metrics = {interval: None for interval in intervals}
# cnt = 0
for interval in interval_elements:
    coderefine_cl_bleu = 0
    coderefine_cl_bleu_diff = 0
    coderefine_cl_edit_sim = 0

    coderefine_cg_bleu = 0
    coderefine_cg_bleu_diff = 0
    coderefine_cg_edit_sim = 0

    codellama_bleu = 0
    codellama_bleu_diff = 0
    codellama_edit_sim = 0

    codegemma_bleu = 0
    codegemma_bleu_diff = 0
    codegemma_edit_sim = 0

    for e in interval_elements[interval]:
        coderefine_cl_truth = e['coderefine_cl_truth']
        coderefine_cl = e['coderefine_cl']
        coderefine_cg_truth = e['coderefine_cg_truth']
        coderefine_cg = e['coderefine_cg']
        base_truth = e['base_truth']
        codellama = e['codellama']
        codegemma = e['codegemma']
        
        coderefine_cl_bleu += sentence_bleu([coderefine_cl_truth.split(" ")], coderefine_cl.split(" "))
        coderefine_cl_bleu_diff += calc_bleu_diff(coderefine_cl_truth, coderefine_cl)
        coderefine_cl_edit_sim += calculate_similarity(coderefine_cl_truth, coderefine_cl)

        coderefine_cg_bleu += sentence_bleu([coderefine_cg_truth.split(" ")], coderefine_cg.split(" "))
        coderefine_cg_bleu_diff += calc_bleu_diff(coderefine_cg_truth, coderefine_cg)
        coderefine_cg_edit_sim += calculate_similarity(coderefine_cg_truth, coderefine_cg)

        codellama_bleu += sentence_bleu([base_truth.split(" ")], codellama.split(" "))
        codellama_bleu_diff += calc_bleu_diff(base_truth, codellama)
        codellama_edit_sim += calculate_similarity(base_truth, codellama)

        codegemma_bleu += sentence_bleu([base_truth.split(" ")], codegemma.split(" "))
        codegemma_bleu_diff += calc_bleu_diff(base_truth, codegemma)
        codegemma_edit_sim += calculate_similarity(base_truth, codegemma)
    total = len(interval_elements[interval])
    if total == 0:
        break
    # cnt += coderefine_em
    interval_metrics[interval] = {"coderefine_cl":{"bleu_diff": coderefine_cl_bleu_diff/total, "bleu": coderefine_cl_bleu/total, "editsim": coderefine_cl_edit_sim/total},
                                  "coderefine_cg":{"bleu_diff": coderefine_cg_bleu_diff/total, "bleu": coderefine_cg_bleu/total, "editsim": coderefine_cg_edit_sim/total},
                                  "codellama":{"bleu_diff": codellama_bleu_diff/total, "bleu": codellama_bleu/total, "editsim": codellama_edit_sim/total},
                                  "codegemma":{"bleu_diff": codegemma_bleu_diff/total, "bleu": codegemma_bleu/total, "editsim": codegemma_edit_sim/total}}

with open("metrics.json", "w") as f:
    json.dump(interval_metrics, f)