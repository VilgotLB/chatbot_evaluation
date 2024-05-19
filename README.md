# Chatbot Evaluation
This repository includes the code, dataset and results of our bachelor thesis about automatic evaluation of chatbots using LLM-as-a-Judge, at Linnaeus University. The authors are Vilgot Lundborg and Yuyao Duan. The program compares three chatbots: Llama 3 70B, ChatGPT4, and Gemini Advanced. The comparison is done in two educational subjects: history and biology. The dataset consists of 30 questions in history and 31 questions in biology, along with the correct answer to each question, and the chatbots answers, three answers per chatbot per question. The evaluation is done automatically using the GPT-4 Turbo API, by asking it to grade each answer from 1 to 5. The results are then displayed using four metrics: average score, which is the average grade accross all answers, average minimum score, which is the average of only the lowest grade to each question, average maximum score, which is the average of only the highest grade to each question, and variability, which is the average standard deviation of the three grades to each question.

## Run the code
1. Install the required libraries using ``pip install openai pandas numpy matplotlib``.
2. Add your OpenAI API key as the ``api_key`` parameter on line 175.
3. Run the script.
