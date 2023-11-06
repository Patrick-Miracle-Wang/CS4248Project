from transformers import pipeline
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

qa_pipeline = pipeline(
  "question-answering",
  model="roberta-squad-large",
  tokenizer="roberta-squad-large"
)

# predictions = qa_pipeline({
#   'context': "The game was played on February 7, 2016 at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
#   'question': "What day was the game played on?"
# })

with open("Preprocessed_dev-v1.1.json", "r") as f:
    questions = json.load(f)

results = []
for question in questions:
    prediction = qa_pipeline({
        "context": question["instruction"],
        "question": question["input"]
    })
    print(question["input"])
    print(prediction)
    print("-------------------------------")
    results.append({"instruction": question["instruction"], "input": question["input"], "output": question["output"], "response": prediction["answer"], "start": prediction["start"], "end": prediction["end"]})

with open("roberta-large.json", "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)