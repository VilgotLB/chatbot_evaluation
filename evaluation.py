from openai import OpenAI
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill

# Creates the prompt to give to the LLM evaluator, which includes the question, correct answer and all nine answers from the chatbots.
def createPrompt (questionNumber, filename):
  # Load entry from csv file
  data=pd.read_csv(filename)
  row = data.iloc[questionNumber-1]
  
  # Separate into question, correct answer and chatbot answers
  question = row.iloc[0]
  correct = row.iloc[1]
  answers = row[2:]

  # Construct prompt
  prompt = ''
  prompt += f'Question: {question}\n\n'
  prompt += f'Correct answer: {correct}\n\n'
  for count, answer in enumerate(answers):
    prompt += f'Answer {count+1}: {answer}\n\n'
  
  return prompt

# Grades all nine answers to the question from 1 to 5.
def gradeQuestion (question_number, filename):
  # Send API request
  prompt = createPrompt(question_number, filename)
  completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
      {"role": "system", "content": "Your task is to grade the answers to a question according to how correct they are with respect to the provided correct answer. There are five possible grades, from 1 to 5, where a higher grade is better. The output shall be in JSON format as an array of objects called \"results\", with each object having two key-value pairs: an \"Answer\" with the answer number, and a \"Grade\" with the given grade."},
      {"role": "user", "content": prompt}
    ],
    response_format={ "type": "json_object" }
  )

  # Parse the response
  response = json.loads(completion.choices[0].message.content)
  grades = [item['Grade'] for item in sorted(response['results'], key=lambda x: x['Answer'])]
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
    std_devs = np.std(reshaped_matrix, axis=2, ddof=1)
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
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.002, round(yval, 2), ha='center', va='bottom')

# Show the results for a subject
def display_results (result, subject):
    # Calculate results
    array = np.array(result)
    avg = average(array)
    avg_min = average_min(array)
    avg_max = average_max(array)
    avg_std = average_stds(array)

    # Print results
    print(subject + ':')
    print(f'Average: {avg}')
    print(f'Average minimum: {avg_min}')
    print(f'Average maximum: {avg_max}')
    print(f'Variability: {avg_std}')

    # Show plots of results
    generate_plot(avg, 'Average', f'Chatbot average scores in {subject}', True)
    generate_plot(avg_min, 'Average minimum', f'Chatbot average minimum scores in {subject}', True)
    generate_plot(avg_max, 'Average maximum', f'Chatbot average maximum scores in {subject}', True)
    generate_plot(avg_std, 'Average standard deviation', f'Chatbot variability in {subject}', False)

# Creates three tables, one for each chatbot, showing their results to each question
def create_tables (data):
    array = np.array(data)

    column_ranges = [(0, 3), (3, 6), (6, 9)]
    grade_ranges = [(4, 5), (3, 4), (2, 3), (1, 2)]
    tables = []

    # For each chatbot
    for start_col, end_col in column_ranges:
        specific_columns = array[:, start_col:end_col]
        averages = specific_columns.mean(axis=1)

        counts = []
        questions = []

        # For each grade range, find the questions that received an average grade in that range.
        for r in grade_ranges:
            if r[0] != 1:
                in_range = np.where((averages > r[0]) & (averages <= r[1]))[0].tolist()
            else:
                in_range = np.where((averages >= r[0]) & (averages <= r[1]))[0].tolist()
            counts.append(len(in_range))
            questions.append(', '.join([f'Q{i + 1}' for i in in_range]))

        table_data = {
            'Average grade': ['4 < x <= 5', '3 < x <= 4', '2 < x <= 3', '1 <= x <= 2'],
            'Count': counts,
            'Questions': questions
        }

        table_df = pd.DataFrame(table_data)
        tables.append((table_df))

    return tables

# Display the tables showing each chatbot's results to each question
def display_tables (data, subject):
    tables = create_tables(data)
    chatbots = ['Llama 3 70B', 'ChatGPT 4', 'Gemini Advanced']
    for iteration, table in enumerate(tables):
        table['Questions'] = table['Questions'].apply(lambda x: fill(x, width=40))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f'{chatbots[iteration]} results in {subject}', fontsize=16)

        ax.xaxis.set_visible(False) 
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

        table_figure = ax.table(cellText=table.values, colLabels=table.columns, cellLoc='center', loc='center')

        table_figure.auto_set_font_size(False)
        table_figure.set_fontsize(12)
        table_figure.scale(4, 4)
        table_figure.auto_set_column_width([0, 1, 2])

        for key, cell in table_figure.get_celld().items():
            if key[1] == 2:
                cell.set_text_props(ha='left')
            if key[0] == 0 or key[1] == -1:
                cell.set_fontsize(14)
                cell.set_text_props(weight='bold')


client = OpenAI(api_key="")


# History grading

history_grades = []
for i in range(1, 31):
  grades = gradeQuestion(i, 'answers/history.csv')
  history_grades.append(grades)


# Biology grading

biology_grades = []
for i in range(1, 32):
  grades = gradeQuestion(i, 'answers/biology.csv')
  biology_grades.append(grades)


# Plots

display_results(history_grades, 'history')
display_results(biology_grades, 'biology')

# Tables

display_tables(history_grades, 'history')
display_tables(biology_grades, 'biology')


plt.show()
