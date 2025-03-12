This script can generate all the results reported in MedVidQA2024@TREC 2024.

Get the judgements of the VCVAL task, topics and ground-truth steps of QFISC task from TREC website.

Put the submission files of teams in a directory, e.g.
```
/path/to/submissions/teamA/run1.json
/path/to/submissions/teamA/run1.json
/path/to/submissions/teamB/run1.json
/path/to/submissions/teamB/run2.json
```

**Install Dependencies**
```
pip install -r requirements.txt
```
Install ```trec_eval``` from here: https://github.com/usnistgov/trec_eval


**Run the evaluations**
```
python medvidqa2024_eval.py \
--path_to_vcval_submissions <PATH_TO_VCVAL_SUBMISSIONS> \
--path_to_qfisc_submissions <path_to_qfisc_submissions> \
--path_to_topics <PATH_TO_TOPICS>\
--path_to_vr_judgement <PATH_TO_VR_JUDGEMENT>\
--path_to_val_judgement <PATH_TO_VAL_JUDGEMENT>\
--path_to_qfisc_gold_steps <PATH_TO_QFISC_GOLD_STEPS>\
--path_to_trec_eval_script <PATH_TO_TREC_EVAL_SCRIPT>\
--path_to_save_results <PATH_TO_SAVE_RESULTS>
```


