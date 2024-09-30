import json
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
import os


def extract_answers_from_dataset(data):
    """
    A function to extract all the possible answers from the dataset dev-v2.0.json
    """
    correct_answers = {}
    question_counter = 1
    
    for item in data['data']:  # Adjusted based on the structure of the JSON file
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                question_key = f"question_{question_counter}"
                question_counter += 1
                
                # Extract correct answers
                answers = list(set(answer['text'] for answer in qa['answers']))
                
                # Extract plausible answers if present
                if 'plausible_answers' in qa and qa['plausible_answers']:
                    plausible_answers = list(set(plausible['text'] for plausible in qa['plausible_answers']))
                    answers.extend(plausible_answers)  # Add plausible answers to the correct answers
                
                # Remove duplicates
                answers = list(set(answers))

                correct_answers[question_key] = answers

    # Write the extracted answers to a JSON file
    with open('first_5000_correct_answers.json', 'w') as output_file:
        json.dump(correct_answers, output_file, indent=4)

def extract_answers_from_gptmini():
    """
    Extracting the answers of the gpt-4o-mini model
    """
    gpt_answers = {}

  
    with open(r"C:\Users\bbaral3\Documents\GitHub\LLM-\batch_output_openai2.jsonl", "r") as f:
        for line in f:
            response = json.loads(line)  # Parse each line as a JSON object
            question_id = response['custom_id']  # Extract the question ID
            model_answer = response['response']['body']['choices'][0]['message']['content']  # Extract the model's answer
            gpt_answers[question_id] = model_answer  # Store the answer in the dictionary

    # Save the answers into a JSON file
    with open('gpt_answers.json', 'w') as output_file:
        json.dump(gpt_answers, output_file, indent=4)

def comparegpt():
    """
    Comparing the correct output with the gpt-4o-mini output
    """
    load_dotenv()

    #correct answers
    with open('first_5000_correct_answers.json', 'r') as f:
        correct_answers = json.load(f)
    #4o mini answers
    with open('gpt_answers.json' , 'r') as f2:
        gpt_answers = json.load(f2)

    scoring_prompt = """
    You are a teacher tasked with determining whether a student’s answer to a question was correct, based on a set of possible correct answers.
    You must only use the provided possible correct answers to determine if the student’s response was correct.
    
    Question: {question}
    Student’s Response: {student_response}
    Possible Correct Answers: {correct_answers}

    Your response should only be a valid JSON as shown below:
    {{
      "explanation" (str): "A short explanation of why the student’s answer was correct or incorrect.",
      "score" (bool): true if the student’s answer was correct, false if it was incorrect
    }}
    Your response:
    """
    
    tasks =[]

    for question_key, corect_answer_list in correct_answers.items():

        student_response = gpt_answers.get(question_key)

        if not student_response:
            continue
        
        # Formatting the scoring prompt for the current question
        prompt =scoring_prompt.format(
            question = question_key,
            student_response = student_response,
            correct_answers = corect_answer_list
        )
        # Creating a task for GPT-4o to evaluate the answer
        task = {
            "custom_id": question_key,
             "method": "POST",
             "url": "/v1/chat/completions",
             "body": {
                "model" : "gpt-4o",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that evaluates student answers."},
                    {"role": "user", "content": prompt}
                ]
             }
        }
        tasks.append(task)

    input_file_path = 'gpt_scoring_tasks.jsonl'

    with open(input_file_path, 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    batch_file = client.files.create(
        file=open(input_file_path, 'rb'),
        purpose='batch'
        )
    print(f"Uploaded batch file: {batch_file.id}")

    batch_job = client.batches.create(
            input_file_id = batch_file.id,
            endpoint = "/v1/chat/completions",
            completion_window="24h"
        )
    print(f"Created batch job: {batch_job.id}")

    # Check the status of the batch job
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        else:
            time.sleep(15)

    print("Batch processing completed.")
    
    
def jsonl_to_json_and_average():

    """
    Converting jsonl batch output to json and calculating the average
    """

    # Define the input file path
    input_file_path = 'gpt_batch_output.jsonl'

    output_file_path = 'gpt-4o-mini-9-30-24-hw3.json'

    correct_count = 0
    total_answers = 0
    processed_results = []

    # Read and process the input file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            processed_result = {
                'custom_id': result['custom_id'],
                'score': None,
                'explanation': None
            }

            # Extract score and explanation from the content
            content = result['response']['body']['choices'][0]['message']['content']
            content = content.replace('\\(', 'LATEX_START').replace('\\)', 'LATEX_END')
             # Remove code block markers if present
            content = remove_code_blocks(content)
            try:
                content_json = json.loads(content)
                processed_result['score'] = content_json.get('score')
                processed_result['explanation'] = content_json.get('explanation')

                if processed_result['score'] is not None:
                    total_answers += 1
                    if processed_result['score'] == True:
                        correct_count += 1
            except json.JSONDecodeError:
                print(f"Error parsing JSON for result: {result['custom_id']}")

            processed_results.append(processed_result)

    # Save the processed results
    with open(output_file_path, 'w') as outfile:
        json.dump(processed_results, outfile, indent=2)

    print(f"Processed results saved to {output_file_path}")

    # Calculate and print the average score
    if total_answers > 0:
        average_score = correct_count / total_answers
        print(f"Total correct answers: {correct_count}")
        print(f"Total answers processed: {total_answers}")
        print(f"Average score for GPT-4o-mini: {average_score:.2%}")
    else:
        print("No valid scores found.")


def remove_code_blocks(text):
    # Use regular expressions to remove code block markers
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    return text.strip()

if __name__ == "__main__":
    # with open('dev-v2.0.json', 'r') as file:
    #     dataset = json.load(file)
    # extract_answers_from_dataset(dataset)
    #comparegpt()
    jsonl_to_json_and_average()



