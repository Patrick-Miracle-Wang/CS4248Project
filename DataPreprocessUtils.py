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
                id = qa["id"]
                formatted_data = {
                    "instruction": context,
                    "input": question,
                    "id":id
                }

                length = len(qa["answers"])
                for i in range(length):
                    key = "output" + str(i)
                    value = qa["answers"][i-1]["text"]
                    formatted_data[key] = value
                result.append(formatted_data)

    #Remove duplicated code
    # reverse dict，filter output
    filterResult = []
    for data in result:
        reversed_data = {}
        for key, value in data.items():
            if value not in reversed_data:
                reversed_data[value] = key

        # reverse to get result
        unique_data = {value: key for key, value in reversed_data.items()}
        filterResult.append(unique_data)
    # Save the transformed data to a new JSON file
    with open(savePath, 'w') as outfile:
        json.dump(filterResult, outfile, indent=4)

    print(savePath)

def convertTestData(testDataPath,resultDataPath):
    # Load the data
    with open(testDataPath, 'r', encoding='utf-8') as file:
        testData = json.load(file)
    with open(resultDataPath, 'r', encoding='utf-8') as file:
        resultData= json.load(file)
    # Add more output to result data

    # Iterate over each entry in resultDataPath
    for resultDataItem in resultData:
        # Find corresponding entry testDataPath
        for testDataItem in testData:
            # Check if 'input' fields are equal
            if testDataItem['input'] == resultDataItem['input']:
                # Start counting additional outputs from 1
                output_counter = 1
                # Iterate over keys in A1 item
                for key in testDataItem.keys():
                    # Check if key starts with 'output' and the output is not the same as in A2
                    if key.startswith('output0') and testDataItem[key] != resultDataItem['output']:
                        # Add the different output to A2 with a new key
                        new_key = f'output{output_counter}'
                        resultDataItem[new_key] = testDataItem[key]
                        # Increment the counter for the next potential output
                        output_counter += 1

    # Save the modified data
    savePath = resultDataPath[0:7] + "new-" + resultDataPath[7:]
    with open(savePath, 'w') as file:
        json.dump(resultData, file, indent=4)


if __name__ == "__main__":
    convertDataFormat("Project/train-v1.1.json","Project/Preprocessed_train-v1.1.json")
    convertDataFormat("Project/dev-v1.1.json","Project/Preprocessed_dev-v1.1.json")
    #convertTestData("Result/t5-3b.json","Project/Preprocessed_dev-v1.1.json")
