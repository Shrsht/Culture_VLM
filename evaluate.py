
import os
import json
import argparse
import re

def extract_answer(response):
    """
    Extracts the answer from the model's response.
    The answer is assumed to be a single uppercase letter.
    """
    match = re.search(r'\b([A-Z])\b', response)
    if match:
        return match.group(1)
    return None

def evaluate(directory):
    """
    Evaluates the model's performance on a set of VQA results.
    """
    correct = 0
    total = 0
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Extract the correct answer
                answer_key = data.get("answer_key")
                if isinstance(answer_key, dict):
                    correct_answer = list(answer_key.values())[0]
                else:
                    correct_answer = answer_key

                # Extract the model's response
                response = data.get("response")
                
                if response:
                    model_answer = extract_answer(response)
                    if model_answer and model_answer == correct_answer:
                        correct += 1
                    total += 1

    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    else:
        print("No JSON files found in the specified directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA model performance.")
    parser.add_argument("directory", type=str, help="The directory containing the JSON result files.")
    args = parser.parse_args()
    evaluate(args.directory)
