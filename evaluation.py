from openai import OpenAI
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creates the prompt to give to the LLM evaluator, which includes the question, correct answer and all nine answers from the chatbots.
def createPrompt (questionNumber, filename):
  data=pd.read_csv(filename)
  row = data.iloc[questionNumber-1]
  
  question = row.iloc[0]
  correct = row.iloc[1]
  answers = row[2:]

  prompt = ''
  prompt += f'Question: {question}\n\n'
  prompt += f'Correct answer: {correct}\n\n'
  for count, answer in enumerate(answers):
    prompt += f'Answer {count+1}: {answer}\n\n'
  
  return prompt

# Grades all nine answers to the question from 1 to 5.
def gradeQuestion (question_number, filename):
  prompt = createPrompt(question_number, filename)
  completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
      {"role": "system", "content": "Your task is to grade the answers to a question according to how correct they are with respect to the provided correct answer. There are five possible grades, from 1 to 5, where a higher grade is better. The output shall be in JSON format as an array of objects called \"results\", with each object having two key-value pairs: an \"Answer\" with the answer number, and a \"Grade\" with the given grade."},
      {"role": "user", "content": prompt}
    ],
    response_format={ "type": "json_object" }
  )
  response = json.loads(completion.choices[0].message.content)
  grades = [item['Grade'] for item in response['results']]
  return grades

# Calculate the average score for each chatbot
def average(matrix):
    reshaped_matrix = matrix.reshape(matrix.shape[0], 3, 3)
    means = np.mean(reshaped_matrix, axis=2)
    mean_means = np.mean(means, axis=0)
    return mean_means

# Calculate the average minimum score for each chatbot
def average_min(matrix):
    reshaped_matrix = matrix.reshape(matrix.shape[0], 3, 3)
    min_values = np.min(reshaped_matrix, axis=2)
    mean_mins = np.mean(min_values, axis=0)
    return mean_mins

# Calculate the average maximum score for each chatbot
def average_max(matrix):
    reshaped_matrix = matrix.reshape(matrix.shape[0], 3, 3)
    max_values = np.max(reshaped_matrix, axis=2)
    mean_maxs = np.mean(max_values, axis=0)
    return mean_maxs

# Calculate the variability for each chatbot (higher values = less consistent)
def average_stds(matrix):
    reshaped_matrix = matrix.reshape(matrix.shape[0], 3, 3)
    std_devs = np.std(reshaped_matrix, axis=2)
    mean_std_devs = np.mean(std_devs, axis=0)
    return mean_std_devs

# Generate a figure (bar chart)
def generate_plot (values, metric, title, fixed_size):
    fig, ax = plt.subplots()
    groups = ['Llama 3 70B', 'ChatGPT 4', 'Gemini Advanced']
    bars = ax.bar(groups, values, color=['red', 'green', 'blue'])
    ax.set_title(title, pad=30)
    ax.set_ylabel(metric)
    if (fixed_size):
        ax.set_ylim(1, 5)
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

# Show the results for a subject
def display_results (result, subject):
    array = np.array(result)
    avg = average(array)
    avg_min = average_min(array)
    avg_max = average_max(array)
    avg_std = average_stds(array)

    print(subject + ':')
    print(f'Average: {avg}')
    print(f'Average minimum: {avg_min}')
    print(f'Average maximum: {avg_max}')
    print(f'Variability: {avg_std}')

    generate_plot(avg, 'Average', f'Chatbot average scores in {subject}', True)
    generate_plot(avg_min, 'Average minimum', f'Chatbot average minimum scores in {subject}', True)
    generate_plot(avg_max, 'Average maximum', f'Chatbot average maximum scores in {subject}', True)
    generate_plot(avg_std, 'Average standard deviation', f'Chatbot variability in {subject}', False)



client = OpenAI(api_key="")


# History

history_grades = []
for i in range(1, 31):
  grades = gradeQuestion(i, 'answers/history.csv')
  history_grades.append(grades)

display_results(history_grades, 'history')


# Biology

biology_grades = []
for i in range(1, 32):
  grades = gradeQuestion(i, 'answers/biology.csv')
  biology_grades.append(grades)

display_results(biology_grades, 'biology')


plt.show()
