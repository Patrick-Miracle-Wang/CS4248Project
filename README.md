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
```shell
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-flan-t5-base.json
```

```shell
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-llama-2-7B-rethink-3-times.json
```

```shell
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-llama-2-7B-rethink.json
```

```shell
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-roberta-base.json
```

```shell
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-roberta-large.json
```

```shell
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-t5-3b-rethink.json
```

```shell
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-t5-3b.json
```

```shell
python3 Project/evaluate-v2.0.py Project/dev-v1.1.json Result/pred-t5-base.json```
```