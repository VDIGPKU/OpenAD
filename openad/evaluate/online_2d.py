"""
version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
"""

from collections import defaultdict
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import math
import pickle


def main(dt_path='result.pkl', gt_path='gt.pkl'):
    """Calculate the AP and AR for OpenAD,
        AP @ IoU=0.5:0.95:0.05 x Clip=0.5:0.9:0.2 for maxinum 500 pred per data
        AR @ IoU=0.5:0.95:0.05 x Clip=0.5:0.9:0.2 for maxinum 100 pred per data

    Parameters
        dt_path (str): user file uploaded
        gt_path (str): ground truth file
    Returns:
            A dictionary with one entry for each metric.
    """

    print('Evaluating OpenAD 2D Results......')

    # separate bbs per image X class
    _bbs = defaultdict(lambda: {"dt": [], "gt": []})

    print("Calculating semantic similarity...")

    with open(dt_path, 'rb') as file:
        dt, text_d, text_feature_d, training_on = pickle.load(file)
    with open(gt_path, 'rb') as file:
        gt, text_g, text_feature_g = pickle.load(file)

    sub_list = [0, 0, 0, 0]
    dataset_se = {
        'av2': (0, 250),
        'kitti': (250, 250 + 309),
        'nuscenes': (250 + 309, 250 + 309 + 134),
        'once': (250 + 309 + 134, 250 + 309 + 134 + 1057),
        'waymo': (250 + 309 + 134 + 1057, 2000),
    }
    for i in range(2000):
        id_cnt = 0
        for dataset in dataset_se.keys():
            if training_on[dataset] and dataset_se[dataset][0] <= i < dataset_se[dataset][1]:
                id_cnt = 1
        for j in range(len(gt[i])):
            box2d = gt[i][j][0:4]
            box3d = gt[i][j][4:11]
            seen = gt[i][j][11:16]
            cls = gt[i][j][-1]
            seen_cnt = 0
            for di, dataset in enumerate(dataset_se.keys()):
                if training_on[dataset]:
                    seen_cnt += int(seen[di])
            if seen_cnt > 0 and id_cnt > 0:
                subc = 0
            elif seen_cnt > 0 and id_cnt == 0:
                subc = 1
            elif seen_cnt == 0 and id_cnt > 0:
                subc = 2
            else:
                subc = 3
            gt[i][j] = box2d + [subc, cls]
            sub_list[subc] += 1
    print(sub_list)
    similarity = text_feature_g @ text_feature_d.T

    print('---Evaluate Setting---')
    print(f'type: 2D, total scene: {len(gt)}, '
          f'total objects: {sum(sub_list)}, '
          f'training on {training_on}, '
          f'in-domain seen objects: {sub_list[0]}, '
          f'out-domain seen objects: {sub_list[1]}, '
          f'in-domain unseen objects: {sub_list[2]}, '
          f'out-domain unseen objects: {sub_list[3]}.')

    for d_idx in range(len(dt)):
        for d in dt[d_idx]:
            _bbs[d_idx, 0]["dt"].append(d)
    for g_idx in range(len(gt)):
        for g in gt[g_idx]:
            _bbs[g_idx, 0]["gt"].append(g)

    sim_mat = (text_g, text_d, similarity)

    # pairwise ious
    clip_thresholds = [0.5, 0.7, 0.9]

    print("Calculating IoU/ATE/ASE...")
    args_list = []
    for k, v in _bbs.items():
        args_list.append((v["dt"], v["gt"], sim_mat))
    with Pool() as pool:
        iou_results = list(tqdm(pool.imap(_compute_ious, args_list), total=len(args_list)))
    _ious = {
        k: v for k, v in zip(_bbs.keys(), iou_results)
    }

    print("Matching and Calculating AP/AR...")

    def _evaluate(iou_threshold, clip_threshold, max_dets):
        # accumulate evaluations
        if clip_threshold == 0.5:
            clip_idx = 0
        elif clip_threshold == 0.7:
            clip_idx = 1
        else:
            clip_idx = 2

        res = {"AP": 0, "total positives": 0, "ATE": 0, "ASE": 0, "subTP": [0, 0, 0, 0], "aps_cnt": 0, "TP": 0}
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id][clip_idx],
                iou_threshold,
                max_dets,
            )
            ev["matched"] = np.array(ev["matched"]).astype(bool)
            evap = _compute_ap_recall(ev["matched"], ev["NP"])
            ev["AP"] = evap["AP"]
            ev["TP"] = evap["TP"]

            # print(img_id, ev["NP"], ev["TP"], ev["subTP"])

            for subi in range(4):
                res["subTP"][subi] += ev["subTP"][subi]
            if ev["TP"] is not None:
                res["TP"] += ev["TP"]
                res["AP"] += ev["AP"]
                res["total positives"] += ev["NP"]
                res["aps_cnt"] += 1
            if ev["ATE"] is not None:
                res["ATE"] += ev["ATE"] * ev["TP"]
                res["ASE"] += ev["ASE"] * ev["TP"]

        res["AP"] = res["AP"] / res["aps_cnt"]
        res["ATE"] = res["ATE"] / res["TP"]
        res["ASE"] = res["ASE"] / res["TP"]
        return [res]

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i * 100 + ci: _evaluate(iou_threshold=i, clip_threshold=ci, max_dets=300)
        for i in iou_thresholds for ci in clip_thresholds
    }

    print('AP Summary', end='')
    for i in iou_thresholds:
        print(f'\tIoU@.{round(i * 100)}', end='')
    print('\n', end='')
    for ci in clip_thresholds:
        print(f'sem@{ci:.2f}', end='')
        for i in iou_thresholds:
            ap = np.mean([x['AP'] for x in full[i * 100 + ci] if x['AP'] is not None])
            print(f'\t{ap:.4f}', end='')
        print('\n', end='')

    print('AR Summary', end='')
    for i in iou_thresholds:
        print(f'\tIoU@.{round(i * 100)}', end='')
    print('\n', end='')
    for ci in clip_thresholds:
        print(f'sem@{ci:.2f}', end='')
        for i in iou_thresholds:
            ar = np.mean([
                x['TP'] / x['total positives'] for x in full[i * 100 + ci] if x['TP'] is not None
            ])
            print(f'\t{ar:.4f}', end='')
        print('\n', end='')

    AP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])
    AR = np.mean([
        x['TP'] / x['total positives'] for k in full for x in full[k] if x['TP'] is not None
    ])
    ASE = np.mean([x['ASE'] for k in full for x in full[k] if x['ASE'] is not None])
    ATE = np.mean([x['ATE'] for k in full for x in full[k] if x['ATE'] is not None])

    new_full = {}
    for k in full.keys():
        if abs(math.modf(k)[0] - 0.9) < 1e-6:
            new_full[k] = full[k]
    full = new_full

    subAR = [None, None, None, None]
    for idx in range(4):
        if sub_list[idx] > 0:
            subAR[idx] = np.mean([x['subTP'][idx] / sub_list[idx] for k in full for x in full[k]])

    overall_metrics = {
        "AP": AP,
        "AR": AR,
        "ATE": ATE,
        "ASE": ASE,
        "AR in-domain seen": subAR[0],
        "AR out-domain seen": subAR[1],
        "AR in-domain unseen": subAR[2],
        "AR out-domain unseen": subAR[3],
    }

    print('---Evaluate Summary---')
    for k, v in overall_metrics.items():
        print(f'{k}: {v:.4f}')

    return overall_metrics


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2, c = a
    return (x2 - x) * (y2 - y)


def _jaccard(a, b, sim_mat):
    xa, ya, x2a, y2a, ca = a
    xb, yb, x2b, y2b, _, cb = b

    similarity = sim_mat[2][sim_mat[0].index(cb)][sim_mat[1].index(ca)]
    if similarity < 0.5:
        return 0, 0, 0, 0

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)

    acenter = ((xa + x2a) / 2, (ya + y2a) / 2)
    bcenter = ((xb + x2b) / 2, (yb + y2b) / 2)
    dis = np.sqrt((acenter[0] - bcenter[0]) ** 2 + (acenter[1] - bcenter[1]) ** 2)

    asize = ((x2a - xa), (y2a - ya))
    bsize = ((x2b - xb), (y2b - yb))
    asAi = min(asize[0], bsize[0]) * min(asize[1], bsize[1])

    return Ai / (Aa + Ab - Ai), dis, 1 - (asAi / (Aa + Ab - asAi)), similarity


def _compute_ious(arg):
    """ compute pairwise ious """
    dt, gt, sim_mat = arg

    ious5 = np.zeros((len(dt), len(gt)))
    ates = np.zeros((len(dt), len(gt)))
    ases = np.zeros((len(dt), len(gt)))
    sims = np.zeros((len(dt), len(gt)))

    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious5[d_idx, g_idx], ates[d_idx, g_idx], ases[d_idx, g_idx], sims[d_idx, g_idx] = _jaccard(d, g,
                                                                                                       sim_mat=sim_mat)

    ious7 = ious5.copy()
    ious7[sims < 0.7] = 0
    ious9 = ious5.copy()
    ious9[sims < 0.9] = 0

    return ((ious5, ates, ases), (ious7, ates, ases), (ious9, ates, ases))


def _evaluate_image(dt, gt, ious3, iou_threshold, max_dets=None, area_range=None):
    """ use COCO's method to associate detections to ground truths """

    dt = dt[:max_dets]
    ious = ious3[0][:max_dets]
    ates = ious3[1][:max_dets]
    ases = ious3[2][:max_dets]

    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] <= _get_area(bb) <= area_range[1])

    gt_ignore = [_is_ignore(g) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    gtm = {}
    dtm = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = ious[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
    ]

    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])

    total_ate = []
    total_ase = []
    sub_tp = [0, 0, 0, 0]
    for g_idx in range(len(gt)):
        if g_idx in gtm.keys():
            total_ate.append(ates[gtm[g_idx]][g_idx])
            total_ase.append(ases[gtm[g_idx]][g_idx])
            sub_tp[gt[g_idx][-2]] += 1
    if len(total_ate) > 0:
        ATE = np.mean(total_ate)
        ASE = np.mean(total_ase)
    else:
        ATE = None
        ASE = None

    return {"matched": matched, "NP": n_gts, "ATE": ATE, "ASE": ASE, "subTP": sub_tp}


def _compute_ap_recall(matched, NP, recall_thresholds=None):
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0
    }
