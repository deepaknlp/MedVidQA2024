import glob
import math
import time
import csv
from tqdm.auto import tqdm
import random
import logging
import argparse
import numpy as np
import subprocess
import os
import json

from prettytable import PrettyTable
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
import evaluate
import language_evaluation

ngram_evaluator = language_evaluation.CocoEvaluator()

# Load metrics
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')

from collections import OrderedDict

logger = logging.getLogger(__name__)

random.seed(42)


def save_json(data, path):
    with open(path, 'w') as wfile:
        json.dump(data, wfile, indent=4)
    print(f"Saved into {path}")


def load_json(path):
    with open(path, 'r') as rfile:
        data = json.load(rfile)
    # print(f"Json file loaded from {path}")
    # print(f"Size : {len(data)}")
    return data


def eval_vr_using_trec_eval(qrels, qid_2_retrieved_videos, path_to_write_qrels,
                            path_to_write_submission, path_to_write_results, path_to_trec_eval_script):
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    with open(path_to_write_qrels, 'w') as wfile:
        question_id = 1
        for qid, rel in qrels.items():
            if qid not in qid_2_retrieved_videos:
                continue
            for video_id, score in rel.items():
                wfile.write(str(question_id) + '\t' + qid + '\t' + video_id + '\t' + str(score) + '\n')
                # wfile.write(str(question_id)+'\t'+qid + '\t0\t' + video_id + '\t' + str(score) + '\n')

            question_id += 1
    with open(path_to_write_submission, 'w') as wfile:
        question_id = 1
        for qid, rel in qid_2_retrieved_videos.items():
            rank = 1
            for video_id, score in rel.items():
                wfile.write(
                    str(question_id) + '\t' + qid + '\t' + video_id + '\t' + str(rank) + '\t' + str(score) + '\tCUR\n')
                rank += 1
            question_id += 1

    result = subprocess.run(
        [path_to_trec_eval_script + "/trec_eval", path_to_write_qrels, path_to_write_submission, "-m", "all_trec"],
        stdout=subprocess.PIPE,
        text=True)
    if result.returncode == 0:
        output = result.stdout
        with open(os.path.join(path_to_write_results), "w") as file:
            file.write(output)
    else:
        print("Error running trec_eval script")
        exit(-1)
    k_values = {5, 10}
    ndcg[f"NDCG"] = 0.0
    _map[f"MAP"] = 0.0
    for k in k_values:
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    with open(os.path.join(path_to_write_results), "r") as rfile:
        for line in rfile:
            metric, _, score = line.strip().split()
            if metric == 'map':
                _map[f"MAP"] = float(score)
            elif metric == 'ndcg':
                ndcg[f"NDCG"] = float(score)
            elif metric == 'P_5':
                precision[f"P@5"] = float(score)
            elif metric == 'P_10':
                precision[f"P@10"] = float(score)
            elif metric == 'recall_5':
                recall[f"Recall@5"] = float(score)
            elif metric == 'recall_10':
                recall[f"Recall@10"] = float(score)

    return _map, recall, precision, ndcg


def eval_val(qrels, n, qid_2_retrieved_videos, rel_type, total_question_count=None):
    required_qrels = {}
    if rel_type == 'All':
        required_qrels = qrels
    else:
        for question_id, video_meta_data in qrels.items():
            new_video_meta_data = {}
            for video_id, segment_data in video_meta_data.items():
                if segment_data['relevancy'] == rel_type:
                    new_video_meta_data[video_id] = segment_data
            required_qrels[question_id] = new_video_meta_data

    k_values = {n}
    for k in k_values:
        ious_4_all_questions = []
        for question_id, video_meta_data in qid_2_retrieved_videos.items():
            count = 0
            predicted_video_and_segments = []
            for video_id, predicted_span in video_meta_data.items():
                predicted_video_and_segments.append([video_id, predicted_span[0], predicted_span[1]])
                count += 1
                if count == k:
                    break
            iou = check_iou(required_qrels[question_id], predicted_video_and_segments)
            ious_4_all_questions.append(iou)
        iou3 = calculate_iou_accuracy(ious_4_all_questions, threshold=0.3, total_question_count=total_question_count)
        iou5 = calculate_iou_accuracy(ious_4_all_questions, threshold=0.5, total_question_count=total_question_count)
        iou7 = calculate_iou_accuracy(ious_4_all_questions, threshold=0.7, total_question_count=total_question_count)
        if total_question_count is not None:
            miou = round((np.sum(ious_4_all_questions) / total_question_count) * 100.0, 2)
        else:
            miou = round((np.mean(ious_4_all_questions)) * 100.0, 2)
        # print(f"K={k}, IoU(0.3) = {iou3}, IoU(0.5) = {iou5}, IoU(0.7) = {iou7}, mIoU = {miou}")
        # print(60 * '*')
        return iou3, iou5, iou7, miou


def read_val_submission_and_evaluate_trec_2024(path_to_read_submission,
                                               path_to_ques2id_to_question,
                                               path_to_read_judgement, path_to_save_results):
    files = glob.glob(path_to_read_submission)

    qrels = load_json(path_to_read_judgement)

    qid2question = {}

    with open(path_to_ques2id_to_question, 'r') as rfile:
        lines = rfile.readlines()
        for line in lines:
            qid2question[line.split('\t')[0]] = line.split('\t')[1].strip()

    n_list = [1, 3, 5, 10]

    x_all = PrettyTable()
    x_all.field_names = ["n", "Team", "RunID", "IoU=0.3", "IoU=0.5", "IoU=0.7", "mIoU"]
    for n in n_list:
        for file in files:
            if file.endswith('.results') or file.endswith('.submission'):
                continue
            team_name = file.split('/')[-2]
            file_name = file.split('/')[-1]

            if team_name == 'BM25':
                continue

            with open(file, 'r') as rfile:
                res = json.load(rfile)
            all_qid_2_retrieved_videos = {}

            qid_list = []

            for item in res:
                preds = [(x['video_id'], x['relevant_score'], x['answer_start_second'], x['answer_end_second']) for x in
                         item['relevant_videos']]
                qid = item['question_id']
                qid_list.append(int(qid.replace('Q', '')))

                sorted_preds = sorted(preds, key=lambda x: preds[0][1])

                for pred in sorted_preds:
                    vid = pred[0]
                    score = pred[1]
                    start_second = pred[2]
                    end_second = pred[3]

                    if qid in all_qid_2_retrieved_videos:
                        all_qid_2_retrieved_videos[qid].update([(vid, (start_second, end_second))])
                    else:

                        all_qid_2_retrieved_videos[qid] = OrderedDict([(vid, (start_second, end_second))])

            for i, x in enumerate(sorted(qid_list)):
                if i + 1 != x:
                    print("Error in submission file.")
                    exit(-1)

            deleted_ques = []
            total_question_count = 0
            for qid, retrieved_videos in list(all_qid_2_retrieved_videos.items()):
                total_question_count += 1
                if qid not in qrels:
                    del all_qid_2_retrieved_videos[qid]
                    deleted_ques.append(qid)
            print("Total questions count: ", total_question_count)

            # print("############## All Questions Evaluation ######################")
            # print("*********** With  Definitely Relevant +  Possibly Relevant ************")
            iou3, iou5, iou7, miou = eval_val(qrels, n, all_qid_2_retrieved_videos, rel_type='All',
                                              total_question_count=total_question_count)

            row = [n, team_name, file_name]
            scores = [iou3, iou5, iou7, miou]
            row.extend(scores)
            x_all.add_row(row)

    print("***********All questions Evaluation************")
    print(x_all.get_string())

    # Convert PrettyTable to CSV
    with open(path_to_save_results, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(x_all.field_names)  # Write header
        writer.writerows(x_all.rows)  # Write rows

    print(f"Data has been written to {path_to_save_results}")
    print(f"Questions for which no DR or PR found: {deleted_ques}")


def calculate_iou_accuracy(ious, threshold, total_question_count=None):
    ### calculate iou based on threshold
    total_size = float(len(ious))
    if total_question_count is not None:
        print(f"total_size: {total_size}")
        print(f"total_question_count: {total_question_count}")

        print("Modifying the total size to total question count as it is not None.")
        total_size = total_question_count
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return round((float(count) / total_size) * 100.0, 2)


def calculate_iou(i0, i1):
    #### compute sample wise iou
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


def check_iou(video_rel, predicted_video_and_span_segments):
    # if predicted_video_and_span_segments[0][0] not in video_rel:
    #     return 0.0
    # else:
    iou_scors = []
    for video_id, video_seg_meta_data in video_rel.items():
        for item in predicted_video_and_span_segments:
            if item[0] == video_id:
                for ans_start, ans_end in zip(video_seg_meta_data['answer_start_seconds'],
                                              video_seg_meta_data['answer_end_seconds']):
                    score = calculate_iou(i0=(item[1], item[2]), i1=(ans_start, ans_end))
                    iou_scors.append(score)

    return max(iou_scors) if len(iou_scors) > 0 else 0.0


def srt_time_to_float_format(string):
    whole = string.strip()
    if len(whole.split(':')) == 3:
        h, m, s = whole.split(':')
        h, m, s = int(h), int(m), int(s)
        total_seconds = h * 60 * 60 + m * 60 + s
    elif len(whole.split(':')) == 2:
        m, s = whole.split(':')
        m, s = int(float(m)), int(float(s))
        total_seconds = m * 60 + s
    else:
        print("Error in srt!")
    total_seconds = str(total_seconds)
    total_seconds = float(total_seconds)
    return total_seconds


######TREC 2024


# Function to calculate time overlap ratio
def calculate_time_overlap(predicted, ground_truth):
    if ":" in str(predicted['step_caption_start']):
        predicted['step_caption_start'] = srt_time_to_float_format(predicted['step_caption_start'])
    if ":" in str(predicted['step_caption_end']):
        predicted['step_caption_end'] = srt_time_to_float_format(predicted['step_caption_end'])
    if ":" in str(ground_truth['step_caption_start']):
        ground_truth['step_caption_start'] = srt_time_to_float_format(ground_truth['step_caption_start'])
    if ":" in str(ground_truth['step_caption_end']):
        ground_truth['step_caption_end'] = srt_time_to_float_format(ground_truth['step_caption_end'])

    pred_start = int(predicted['step_caption_start'])
    pred_end = int(predicted['step_caption_end'])
    gt_start = int(ground_truth['step_caption_start'])
    gt_end = int(ground_truth['step_caption_end'])

    iou = calculate_iou([pred_start, pred_end], [gt_start, gt_end])

    return iou


def calculate_relaxed_time_overlap(predicted, ground_truth, extend_value=1):
    if ":" in str(predicted['step_caption_start']):
        predicted['step_caption_start'] = srt_time_to_float_format(predicted['step_caption_start'])
    if ":" in str(predicted['step_caption_end']):
        predicted['step_caption_end'] = srt_time_to_float_format(predicted['step_caption_end'])
    if ":" in str(ground_truth['step_caption_start']):
        ground_truth['step_caption_start'] = srt_time_to_float_format(ground_truth['step_caption_start'])
    if ":" in str(ground_truth['step_caption_end']):
        ground_truth['step_caption_end'] = srt_time_to_float_format(ground_truth['step_caption_end'])

    pred_start = int(predicted['step_caption_start'])
    pred_end = int(predicted['step_caption_end'])
    gt_start = int(ground_truth['step_caption_start'])
    gt_end = int(ground_truth['step_caption_end'])

    iou = calculate_iou([pred_start, pred_end], [gt_start, gt_end])
    if iou == 0:
        window_scores = []
        temp_pred_start = pred_start - extend_value
        temp_pred_end = pred_end - extend_value
        iou = calculate_iou([temp_pred_start, temp_pred_end], [gt_start, gt_end])
        window_scores.append(iou)

        temp_pred_start = pred_start + extend_value
        temp_pred_end = pred_end - extend_value
        iou = calculate_iou([temp_pred_start, temp_pred_end], [gt_start, gt_end])
        window_scores.append(iou)

        temp_pred_start = pred_start - extend_value
        temp_pred_end = pred_end + extend_value
        iou = calculate_iou([temp_pred_start, temp_pred_end], [gt_start, gt_end])
        window_scores.append(iou)

        temp_pred_start = pred_start + extend_value
        temp_pred_end = pred_end + extend_value
        iou = calculate_iou([temp_pred_start, temp_pred_end], [gt_start, gt_end])
        window_scores.append(iou)

        iou = max(window_scores)
    return iou


# Function to calculate alignment score using both time overlap and ROUGE-L similarity
def calculate_alignment_score(predicted_step, ground_truth_step, overlap_weight=0.5, rouge_weight=0.5):
    time_overlap = calculate_time_overlap(predicted_step, ground_truth_step)

    rouge_results = rouge.compute(predictions=[predicted_step['step_caption']],
                                  references=[ground_truth_step['step_caption']])
    rouge_l_similarity = rouge_results['rougeL']  # ROUGE-L score

    alignment_score = overlap_weight * time_overlap + rouge_weight * rouge_l_similarity
    return alignment_score


# Function to align predicted steps to ground truth steps while maintaining order
def align_steps(predicted_steps, ground_truth_steps, overlap_threshold=0.3):
    TP, FP, FN = 0, 0, 0
    matched_pairs = []

    gt_index = 0

    for pred_step in predicted_steps:
        best_score = -1
        best_gt_index = -1

        for i in range(gt_index, len(ground_truth_steps)):
            gt_step = ground_truth_steps[i]
            score = calculate_alignment_score(pred_step, gt_step)

            if score > best_score and score >= overlap_threshold:
                best_score = score
                best_gt_index = i

        if best_gt_index != -1:
            TP += 1
            matched_pairs.append((pred_step, ground_truth_steps[best_gt_index]))
            gt_index = best_gt_index + 1
        else:
            FP += 1

    FN = len(ground_truth_steps) - TP

    return TP, FP, FN, len(ground_truth_steps), matched_pairs,


# Function to calculate precision, recall, and F-score
def calculate_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f_score


def process_and_sort_step_timestamp(step_data_list):
    for step_data in step_data_list:
        if ":" in str(step_data['step_caption_start']):
            step_data['step_caption_start'] = srt_time_to_float_format(step_data['step_caption_start'])
        if ":" in str(step_data['step_caption_end']):
            step_data['step_caption_end'] = srt_time_to_float_format(step_data['step_caption_end'])

        step_data['step_caption_start'] = int(step_data['step_caption_start'])
        step_data['step_caption_end'] = int(step_data['step_caption_end'])
    sorted_step_data_list = sorted(step_data_list, key=lambda x: x['step_caption_start'])
    return sorted_step_data_list


def get_ngram_matching_scores(ground_truth_data, submitted_data):
    id2ground_truth = {}
    id2submitted_data = {}
    for g_item in ground_truth_data:
        assert g_item['sample_id'] not in id2ground_truth
        id2ground_truth[g_item['sample_id']] = g_item

    for s_item in submitted_data:
        id2submitted_data[s_item['sample_id']] = s_item

    results = {}
    count = 0
    limit = 5
    for sample_id, s_sample_data in tqdm(id2submitted_data.items()):
        count += 1
        # if limit==count:
        #     break
        if type(id2ground_truth[sample_id]['steps_list'][0]) == list:
            score_list = []
            max_score_position = -1
            max_rl_score = 0.0

            for idx, gt_step_list in enumerate(id2ground_truth[sample_id]['steps_list']):

                s_sample_data['steps_list'] = process_and_sort_step_timestamp(s_sample_data['steps_list'])
                gt_step_list = process_and_sort_step_timestamp(gt_step_list)

                predicted_steps = [item['step_caption'] for item in s_sample_data['steps_list']]
                gt_steps = [item['step_caption'] for item in gt_step_list]

                predicted_steps = ' '.join(predicted_steps)
                gt_steps = ' '.join(gt_steps)

                ngram_results = ngram_evaluator.run_evaluation([predicted_steps], [gt_steps])

                rouge_scores = rouge.compute(predictions=[predicted_steps], references=[gt_steps])
                rouge_l_score = rouge_scores['rougeL']
                bert_scores = bertscore.compute(predictions=[predicted_steps], references=[gt_steps], lang="en",
                                                batch_size=8)
                del bert_scores['precision']
                del bert_scores['recall']
                del bert_scores['hashcode']
                bert_scores['bert_score'] = bert_scores['f1'][0]
                del bert_scores['f1']
                ngram_results.update(rouge_scores)
                ngram_results.update(bert_scores)

                score_list.append(ngram_results)
                if rouge_l_score > max_rl_score:
                    max_rl_score = rouge_l_score
                    max_score_position = idx

            results[sample_id] = score_list[max_score_position]

        else:

            s_sample_data['steps_list'] = process_and_sort_step_timestamp(s_sample_data['steps_list'])
            id2ground_truth[sample_id]['steps_list'] = process_and_sort_step_timestamp(
                id2ground_truth[sample_id]['steps_list'])

            predicted_steps = [item['step_caption'] for item in s_sample_data['steps_list']]
            gt_steps = [item['step_caption'] for item in id2ground_truth[sample_id]['steps_list']]

            predicted_steps = ' '.join(predicted_steps)
            gt_steps = ' '.join(gt_steps)

            ngram_results = ngram_evaluator.run_evaluation([predicted_steps], [gt_steps])
            rouge_scores = rouge.compute(predictions=[predicted_steps], references=[gt_steps])
            bert_scores = bertscore.compute(predictions=[predicted_steps], references=[gt_steps], lang="en",
                                            batch_size=8)
            del bert_scores['precision']
            del bert_scores['recall']
            del bert_scores['hashcode']
            bert_scores['bert_score'] = bert_scores['f1'][0]
            del bert_scores['f1']

            ngram_results.update(rouge_scores)
            ngram_results.update(bert_scores)
            results[sample_id] = ngram_results
    metrics_sum = {}
    count = len(id2ground_truth)

    for sample_id, metrics in results.items():
        # Loop through each metric for the sample
        for key, value in metrics.items():
            if key == 'matched_pairs':
                continue

            if key not in metrics_sum:
                metrics_sum[key] = 0
            metrics_sum[key] += value  # Add the metric value to the running total

    # Calculate the average for each metric
    averages = {key: metrics_sum[key] / count for key in metrics_sum}
    return averages


def get_overlap_scores(matched_pairs, gt_len, extend_value):
    ious_4_all_pairs = []

    for match_pair in matched_pairs:
        iou = calculate_relaxed_time_overlap(match_pair[0], match_pair[1], extend_value=extend_value)
        ious_4_all_pairs.append(iou)

    for i in range(gt_len - len(matched_pairs)):
        ious_4_all_pairs.append(0)

    assert len(ious_4_all_pairs) == gt_len

    iou3 = calculate_iou_accuracy(ious_4_all_pairs, threshold=0.3)
    iou5 = calculate_iou_accuracy(ious_4_all_pairs, threshold=0.5)
    iou7 = calculate_iou_accuracy(ious_4_all_pairs, threshold=0.7)
    miou = round((np.mean(ious_4_all_pairs)) * 100.0, 2)

    results = {'overlap_iou3': iou3, 'overlap_iou5': iou5, 'overlap_iou7': iou7, 'overlap_miou': miou}

    return results


def compute_metric_for_qfisc(ground_truth_data, submitted_data, rt, ev):
    id2ground_truth = {}
    id2submitted_data = {}
    for g_item in ground_truth_data:
        assert g_item['sample_id'] not in id2ground_truth
        id2ground_truth[g_item['sample_id']] = g_item

    for s_item in submitted_data:
        id2submitted_data[s_item['sample_id']] = s_item

    results = {}
    limit = 5
    count = 0
    for sample_id, s_sample_data in tqdm(id2submitted_data.items()):
        count += 1
        # if count==limit:
        #     break
        if type(id2ground_truth[sample_id]['steps_list'][0]) == list:
            score_list = []
            max_score_position = -1
            max_f_score = 0.0

            for idx, gt_step_list in enumerate(id2ground_truth[sample_id]['steps_list']):
                TP, FP, FN, gt_len, matched_pairs = align_steps(s_sample_data['steps_list'], gt_step_list,
                                                                overlap_threshold=rt)
                overlap_results = get_overlap_scores(matched_pairs, gt_len, extend_value=ev)
                # Calculate precision, recall, and F-score
                precision, recall, f_score = calculate_metrics(TP, FP, FN)
                temp_res = {'precision': precision,
                            'recall': recall,
                            'f-score': f_score,
                            'matched_pairs': matched_pairs}
                temp_res.update(overlap_results)
                score_list.append(temp_res)
                if f_score > max_f_score:
                    max_f_score = f_score
                    max_score_position = idx

            results[sample_id] = score_list[max_score_position]

        else:
            TP, FP, FN, gt_len, matched_pairs = align_steps(s_sample_data['steps_list'],
                                                            id2ground_truth[sample_id]['steps_list'],
                                                            overlap_threshold=rt)
            # Calculate precision, recall, and F-score
            precision, recall, f_score = calculate_metrics(TP, FP, FN)
            overlap_results = get_overlap_scores(matched_pairs, gt_len, extend_value=ev)
            temp_res = {'precision': precision,
                        'recall': recall,
                        'f-score': f_score,
                        'matched_pairs': matched_pairs}
            temp_res.update(overlap_results)
            results[sample_id] = temp_res
    metrics_sum = {}
    count = len(id2ground_truth)
    for sample_id, metrics in results.items():
        # Loop through each metric for the sample
        for key, value in metrics.items():
            if key == 'matched_pairs':
                continue
            if key not in metrics_sum:
                metrics_sum[key] = 0
            metrics_sum[key] += value  # Add the metric value to the running total

    # Calculate the average for each metric
    averages = {key: metrics_sum[key] / count for key in metrics_sum}

    return averages


def read_qfisc_submissions_and_evaluate_trec_2024(path_to_read_submissions,
                                                  path_to_ground_truth_annotations,
                                                  path_to_write_result_file,
                                                  rouge_thresholds=[0.4],
                                                  extension_values=[3]):
    run_files = glob.glob(os.path.join(path_to_read_submissions, '*/*'))
    ground_truth_data = load_json(path_to_ground_truth_annotations)

    sampled_id2_submissions = {}
    for g_item in ground_truth_data:
        assert g_item['sample_id'] not in sampled_id2_submissions
        sampled_id2_submissions[g_item['sample_id']] = []
        if type(g_item['steps_list'][0]) == list:
            for temp_g_item in g_item['steps_list']:
                sampled_id2_submissions[g_item['sample_id']].append({'team_name': 'NLM',
                                                                     'run_name': 'Ground Truth',
                                                                     'steps_list': temp_g_item,
                                                                     "question": g_item['question'],
                                                                     "video_id": g_item['video_id'],
                                                                     "video_length": g_item['video_length'],
                                                                     "segment_start": g_item['segment_start'],
                                                                     "segment_end": g_item['segment_end'],
                                                                     })
        else:
            sampled_id2_submissions[g_item['sample_id']].append({'team_name': 'NLM',
                                                                 'run_name': 'Ground Truth',
                                                                 'steps_list': g_item['steps_list'],
                                                                 "question": g_item['question'],
                                                                 "video_id": g_item['video_id'],
                                                                 "video_length": g_item['video_length'],
                                                                 "segment_start": g_item['segment_start'],
                                                                 "segment_end": g_item['segment_end'],
                                                                 })

    final_result_list = []
    for run_file in run_files:
        with open(run_file, 'r') as f:
            run_name = os.path.basename(run_file)
            team_name = run_file.split('/')[-2]
            run_data = json.load(f)
            for s_item in run_data:
                assert s_item['sample_id'] in sampled_id2_submissions

                refined_steps_list = []
                for step_list in s_item['steps_list']:
                    if step_list['step_caption_start'] is None or step_list['step_caption_end'] is None:
                        print(team_name + '\t' + run_name)
                        continue
                    refined_steps_list.append(step_list)

                sampled_id2_submissions[s_item['sample_id']].append({'team_name': team_name,
                                                                     'run_name': run_name,
                                                                     'steps_list': refined_steps_list,
                                                                     "question": s_item['question'],
                                                                     "video_id": s_item['video_id'],
                                                                     "video_length": s_item['video_length'],
                                                                     "segment_start": s_item['segment_start'],
                                                                     "segment_end": s_item['segment_end'],
                                                                     })

            print(f"Sample size before process: {len(run_data)}")

            for i, s_item in enumerate(run_data):
                refined_steps_list = []
                for step_list in s_item['steps_list']:
                    if step_list['step_caption_start'] is None or step_list['step_caption_start'] == 'null' or \
                            step_list['step_caption_end'] is None or step_list['step_caption_end'] == 'null':
                        print(team_name + '\t' + run_name)
                        continue
                    refined_steps_list.append(step_list)
                if len(refined_steps_list) != 0:
                    s_item['steps_list'] = refined_steps_list
                elif len(refined_steps_list) == 0:
                    run_data.pop(i)
            print(f"Sample size after process: {len(run_data)}")
            meta_result = {'team_name': team_name, 'run_name': run_name}
            n_gram_eval_done = False
            for rt in rouge_thresholds:
                for ev in extension_values:
                    res1 = compute_metric_for_qfisc(ground_truth_data, run_data, rt=rt, ev=ev)
                    meta_result['rouge_threshold'] = rt
                    meta_result['extension_value'] = ev
                    meta_result.update(res1)
                    if not n_gram_eval_done:
                        res2 = get_ngram_matching_scores(ground_truth_data, run_data)
                        n_gram_eval_done = True
                        meta_result.update(res2)
                    final_result_list.append(meta_result)

    save_json(final_result_list, path_to_write_result_file.replace('.csv', '.json'))

    with open(path_to_write_result_file, mode='w', newline='') as file:
        # Create a CSV DictWriter object
        writer = csv.DictWriter(file, fieldnames=final_result_list[0].keys())

        # Write the header (column names)
        writer.writeheader()

        # Write the data
        writer.writerows(final_result_list)

    print(f"Data has been written to {path_to_write_result_file}")


def read_vcval_submission_and_evaluate_trec_2024(path_to_read_submission,
                                                 path_to_ques2id_to_question,
                                                 path_to_read_judgement,
                                                 path_to_trec_eval_script, path_to_save_results):
    files = glob.glob(path_to_read_submission)
    qrels = load_json(path_to_read_judgement)

    qid2question = {}

    with open(path_to_ques2id_to_question, 'r') as rfile:
        lines = rfile.readlines()
        for line in lines:
            qid2question[line.split('\t')[0]] = line.split('\t')[1].strip()

    x_all = PrettyTable()
    x_all.field_names = ["Team", "RunID", "MAP", "R@5", "R@10", "P@5", "P@10", "nDCG"]

    for file in files:
        if file.endswith('.results') or file.endswith('.submission'):
            continue
        team_name = file.split('/')[-2]
        file_name = file.split('/')[-1]

        with open(file, 'r') as rfile:
            res = json.load(rfile)
        all_qid_2_retrieved_videos = {}

        qid_list = []
        for item in res:
            preds = [(x['video_id'], x['relevant_score']) for x in item['relevant_videos']]
            qid = item['question_id']
            qid_list.append(int(qid.replace('Q', '')))

            sorted_preds = sorted(preds, key=lambda x: preds[0][1])

            for pred in sorted_preds:
                vid = pred[0]
                score = pred[1]

                if qid in all_qid_2_retrieved_videos:
                    all_qid_2_retrieved_videos[qid].update([(vid, score)])
                else:
                    all_qid_2_retrieved_videos[qid] = OrderedDict([(vid, score)])

        for i, x in enumerate(sorted(qid_list)):
            if i + 1 != x:
                print("Error in submission file.")
                exit(-1)
        path_to_write_qrels = path_to_read_judgement.replace('.json', '-all-questions-trec-format.qrels')
        path_to_write_submission = file + '-all-questions-trec-format.submission'
        path_to_write_results = file + '-all-questions-trec-eval.results'
        _map, recall, precision, ndcg = eval_vr_using_trec_eval(qrels, all_qid_2_retrieved_videos,
                                                                path_to_write_qrels=path_to_write_qrels,
                                                                path_to_write_submission=path_to_write_submission,
                                                                path_to_write_results=path_to_write_results,
                                                                path_to_trec_eval_script=path_to_trec_eval_script)

        row = [team_name, file_name]
        scores = []

        scores.append(_map[f"MAP"])
        scores.append(recall[f"Recall@{5}"])
        scores.append(recall[f"Recall@{10}"])
        scores.append(precision[f"P@{5}"])
        scores.append(precision[f"P@{10}"])
        scores.append(ndcg[f"NDCG"])
        row.extend(scores)
        x_all.add_row(row)

    print("***********All questions Evaluation************")
    print(x_all.get_string())
    # Convert PrettyTable to CSV

    with open(path_to_save_results, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(x_all.field_names)  # Write header
        writer.writerows(x_all.rows)  # Write rows

    print(f"Data has been written to {path_to_save_results}")


def float_to_srt_time_format(d: float) -> str:
    """Convert decimal durations into proper srt format.
    :rtype: str
    :returns:
        SubRip Subtitle (str) formatted time duration.
    float_to_srt_time_format(3.89) -> '00:00:03'  #### not considering the micro-second
    """
    fraction, whole = math.modf(d)
    time_fmt = time.strftime("%H:%M:%S", time.gmtime(whole))
    ms = f"{fraction:.3f}".replace("0.", "")
    return time_fmt


parser = argparse.ArgumentParser()

parser.add_argument('--path_to_vcval_submissions', type=str)
parser.add_argument('--path_to_qfisc_submissions', type=str)
parser.add_argument('--path_to_topics', type=str)
parser.add_argument('--path_to_vr_judgement', type=str)
parser.add_argument('--path_to_val_judgement', type=str)
parser.add_argument('--path_to_qfisc_gold_steps', type=str)
parser.add_argument('--path_to_trec_eval_script', type=str)
parser.add_argument('--path_to_save_results', type=str)

configs = parser.parse_args()

read_vcval_submission_and_evaluate_trec_2024(
    path_to_read_submission=configs.path_to_vcval_submissions,
    path_to_ques2id_to_question=configs.path_to_topics,
    path_to_read_judgement=configs.path_to_vr_judgement,
    path_to_trec_eval_script=configs.path_to_trec_eval_script,
    path_to_save_results=os.path.join(configs.path_to_save_results, 'vr-results.csv')
)

# #
read_val_submission_and_evaluate_trec_2024(
    path_to_read_submission=configs.path_to_vcval_submissions,
    path_to_ques2id_to_question=configs.path_to_topics,
    path_to_read_judgement=configs.path_to_val_judgement,
    path_to_save_results=os.path.join(configs.path_to_save_results, 'val-results.csv')
)

read_qfisc_submissions_and_evaluate_trec_2024(
    path_to_read_submission=configs.path_to_qfisc_submissions,
    path_to_ground_truth_annotations=configs.path_to_qfisc_gold_steps,
    path_to_write_result_file=os.path.join(configs.path_to_save_results, 'qfisc-results.csv'),
    rouge_thresholds=[0.4], extension_values=[3]
)
