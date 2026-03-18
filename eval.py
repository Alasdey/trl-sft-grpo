import re


RELATIONS = ["CAUSE", "PRECONDITION", "coreference", "subevent", "BEFORE", "BEGINS-ON", "CONTAINS", "ENDS-ON", "OVERLAP", "SIMULTANEOUS"]

def compute_f1(pred_list, gold_list):
    npred = sum([len(set(pred)) for pred in pred_list])
    ngold = sum([len(set(gold)) for gold in gold_list])
    if npred==0 and ngold==0:
        return 0.05

    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(pred_list, gold_list):
        tp += len(set(pred) & set(gold))
        fp += len(set(pred) - set(gold))
        fn += len(set(gold) - set(pred))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

def rel_extract(text):
    """Extract best as possible the relations in the text."""
    flag_pattern = r":((?:\s*?[et][0-9]{1,3})*|\s*?none);?"
    fps = [r + flag_pattern for r in RELATIONS]
    res = []
    for pattern in fps:
        res.append([])
        rel_seg = re.findall(pattern, text)
        for segment in rel_seg:
            res[-1] = re.findall(r"[et][0-9]+", segment)
    return res

def accuracy_reward(pred, gold):
    gold = rel_extract(gold)
    pred = rel_extract(pred)
    return compute_f1(pred, gold)