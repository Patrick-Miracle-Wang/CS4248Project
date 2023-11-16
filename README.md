# CS4248Project

# DataPreprocessing

1.Use DataPreprocessUtils.py to get question-answer paris file

run DataPostprocessing.py main()

# Train Model

run the shell in train_and_eval/t5, train_and_eval/llama-2, train_and_eval/roberta

# Evaluate

1.Use DataPostprocessing.py generate the pred file

run DataPostprocessing.py main()

2.Use evaluate.py

python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-flan-t5-base.json
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-llama-2-7B-rethink-3-times.json
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-llama-2-7B-rethink.json
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-roberta-base.json
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-roberta-large.json
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-t5-3b-rethink.json
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-t5-3b.json
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-t5-base.json