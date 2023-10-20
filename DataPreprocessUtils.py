import json

def convertDataFormat(dataPath,savePath):
    # Read the input JSON file
    with open(dataPath, 'r') as file:
        input_data = json.load(file)

    # Convert the input data to the desired format
    result = []

    for data_entry in input_data["data"]:
        for paragraph in data_entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]["text"]  # Assuming there's at least one answer
                formatted_data = {
                    "instruction": context,
                    "input": question,
                    "output": answer
                }
                result.append(formatted_data)

    # Save the transformed data to a new JSON file
    with open(savePath, 'w') as outfile:
        json.dump(result, outfile, indent=4)

    print("Data saved to preprocessed_train-v1.1.json")


if __name__ == "__main__":
    convertDataFormat("Project/train-v1.1.json","Project/Preprocessed_train-v1.1.json")
    convertDataFormat("Project/dev-v1.1.json","Project/Preprocessed_dev-v1.1.json")
