"""
version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
"""

from collections import defaultdict
import numpy as np
import clip
import torch
from tqdm import tqdm


def get_2d_summary(groundtruth_bbs, detected_bbs):
    """Calculate the AP and AR for OpenAD,
        AP @ IoU=0.5:0.95:0.05 x Clip=0.5:0.9:0.2 for maxinum 500 pred per data
        AR @ IoU=0.5:0.95:0.05 x Clip=0.5:0.9:0.2 for maxinum 100 pred per data

    Parameters
        ----------
            groundtruth_bbs : list
                A list representing the ground-truth bounding boxes.
                (list)[
                    N_images * (list)[
                        N_bboxes * (list)[ (float)x1, y1, x2, y2, (str)c ]
                    ]
                ]
            detected_bbs : list
                A list representing the detected bounding boxes.
                (list)[
                    N_images * (list)[
                        N_bboxes * (list)[ (float)x1, y1, x2, y2, (str)c ]
                    ]
                ]
    Returns:
            A dictionary with one entry for each metric.
    """

    print('Evaluating OpenAD 2D Results......')

    assert len(groundtruth_bbs) == len(detected_bbs)
    for i in range(len(detected_bbs)):
        if len(detected_bbs[i]) > 500:
            detected_bbs[i] = detected_bbs[i][:500]
            print(f'Number of predicted objects exceeds 500, only the first 500 will be calculated. (data index {i})')

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    clip_thresholds = [0.5, 0.7, 0.9]

    _ious = {
        ci: {k: _compute_ious(**v, clip_thr=ci) for k, v in _bbs.items()}
        for ci in clip_thresholds
    }

    def _evaluate(iou_threshold, clip_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"matched": [], "NP": [], "ATE": [], "ASE": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[clip_threshold][img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            acc = _evals[class_id]
            acc["matched"].append(ev["matched"])
            # print(acc["matched"])
            acc["NP"].append(ev["NP"])
            if ev["ATE"] is not None:
                acc["ATE"].append(ev["ATE"])
                acc["ASE"].append(ev["ASE"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                "ATE": np.mean(ev["ATE"]) if len(ev["ATE"]) > 0 else None,
                "ASE": np.mean(ev["ASE"]) if len(ev["ASE"]) > 0 else None,
                **_compute_ap_recall(ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i * 100 + ci: _evaluate(iou_threshold=i, clip_threshold=ci, max_dets=500, area_range=(0, np.inf))
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

    max_det100 = {
        i * 100 + ci: _evaluate(iou_threshold=i, clip_threshold=ci, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds for ci in clip_thresholds
    }

    print('AR Summary', end='')
    for i in iou_thresholds:
        print(f'\tIoU@.{round(i * 100)}', end='')
    print('\n', end='')
    for ci in clip_thresholds:
        print(f'sem@{ci:.2f}', end='')
        for i in iou_thresholds:
            ar = np.mean([
                x['TP'] / x['total positives'] for x in max_det100[i * 100 + ci] if x['TP'] is not None
            ])
            print(f'\t{ar:.4f}', end='')
        print('\n', end='')

    AP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])
    AR100 = np.mean([
        x['TP'] / x['total positives'] for k in max_det100 for x in max_det100[k] if x['TP'] is not None
    ])
    ASE = np.mean([x['ASE'] for k in full for x in full[k] if x['ASE'] is not None])
    ATE = np.mean([x['ATE'] for k in full for x in full[k] if x['ATE'] is not None])

    print(f'AP = {AP}')
    print(f'AR = {AR100}')
    print(f'ATE = {ATE}')
    print(f'ASE = {ASE}')

    return {
        "AP": AP,
        "AR": AR100,
        "ATE": ATE,
        "ASE": ASE,
    }


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})

    if type(dt[0][0][-1]) is str:
        clip_model, _ = clip.load("ViT-L/14@336px")
        clip_model.cuda()

        text = []
        for d_idx in range(len(dt)):
            for d in dt[d_idx]:
                if len(d[-1]) > 75:
                    d[-1] = d[-1][:75]
                text.append(clip.tokenize(['a ' + d[-1]]))
        for g_idx in range(len(gt)):
            for g in gt[g_idx]:
                if len(g[-1]) > 75:
                    g[-1] = g[-1][:75]
                text.append(clip.tokenize(['a ' + g[-1]]))

        text = torch.stack(text, dim=0).squeeze(1).cuda()
        text_list = torch.split(text, 300, dim=0)
        text_features_list = []
        print('forward clip')
        for i in tqdm(text_list):
            with torch.no_grad():
                text_features = clip_model.encode_text(i)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                text_features_list.append(text_features)
        text_features = torch.cat(text_features_list, dim=0)

        text_idx = 0
        for d_idx in range(len(dt)):
            for d in dt[d_idx]:
                d[-1] = text_features[text_idx]
                text_idx += 1
                bb_info[d_idx, 0]["dt"].append(d)
        for g_idx in range(len(gt)):
            for g in gt[g_idx]:
                g[-1] = text_features[text_idx]
                text_idx += 1
                bb_info[g_idx, 0]["gt"].append(g)

        return bb_info

    elif type(dt[0][0][-1]) is torch.Tensor:
        for d_idx in range(len(dt)):
            for d in dt[d_idx]:
                bb_info[d_idx, 0]["dt"].append(d)
        for g_idx in range(len(gt)):
            for g in gt[g_idx]:
                bb_info[g_idx, 0]["gt"].append(g)

        return bb_info

    else:
        raise ValueError(f'box[-1] must be str. get {type(dt[0][0][-1])}.')


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2, c = a
    return (x2 - x) * (y2 - y)


def _jaccard(a, b, clip_thr):
    xa, ya, x2a, y2a, ca = a
    xb, yb, x2b, y2b, cb = b

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


    similarity = ca @ cb.t()

    acenter = ((xa + x2a) / 2, (ya + y2a) / 2)
    bcenter = ((xb + x2b) / 2, (yb + y2b) / 2)
    dis = np.sqrt((acenter[0] - bcenter[0]) ** 2 + (acenter[1] - bcenter[1]) ** 2)

    asize = ((x2a-xa), (y2a-ya))
    bsize = ((x2b-xb), (y2b-yb))
    asAi = min(asize[0], bsize[0]) * min(asize[1], bsize[1])

    if similarity > clip_thr:
        return Ai / (Aa + Ab - Ai), dis, 1 - (asAi / (Aa + Ab - asAi))
    else:
        return 0, 0, 0


def _compute_ious(dt, gt, clip_thr):
    """ compute pairwise ious """

    ious = np.zeros((len(dt), len(gt)))
    ates = np.zeros((len(dt), len(gt)))
    ases = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx], ates[d_idx, g_idx], ases[d_idx, g_idx] = _jaccard(d, g, clip_thr=clip_thr)
    return (ious, ates, ases)


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
    for g_idx in range(len(gt)):
        if g_idx in gtm.keys():
            total_ate.append(ates[gtm[g_idx]][g_idx])
            total_ase.append(ases[gtm[g_idx]][g_idx])
    if len(total_ate) > 0:
        ATE = np.mean(total_ate)
        ASE = np.mean(total_ase)
    else:
        ATE = None
        ASE = None

    return {"matched": matched, "NP": n_gts, "ATE": ATE, "ASE": ASE}


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
    n_recalls = len(recall_thresholds)

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
