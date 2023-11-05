# Author Oliver Li
# Date 2023/11/5 16:38


import json

def getPredFileFromResult(filePath,datasetPath):
    with open(filePath,'r',) as file:
        data = json.load(file)

    with open(datasetPath, 'r') as file1:
        dataset = json.load(file1)

    ids = getIds(dataset)

    predData = {}
    for index in range(len(data)):
        predData[ids[index]] = data[index]["response"]

    savePath = filePath[0:7] + "pred-" + filePath[7:]
    with open(savePath,'w') as file1:
        json.dump(predData, file1, indent=4)

def getIds(dataset):
    ids = []
    for article in dataset['data']:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                ids.append(qid)

    return ids

if __name__ == "__main__":
    #getPredFileFromResult("Result/flan-t5-base.json", "Project/dev-v1.1.json")
    #getPredFileFromResult("Result/llama-2-7B.json", "Project/dev-v1.1.json")
    #getPredFileFromResult("Result/roberta-base.json", "Project/dev-v1.1.json")
    #getPredFileFromResult("Result/roberta-large.json", "Project/dev-v1.1.json")
    #getPredFileFromResult("Result/t5-3b.json","Project/dev-v1.1.json")
    getPredFileFromResult("Result/t5-base.json","Project/dev-v1.1.json")
