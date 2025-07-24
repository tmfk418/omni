import textwrap

def build_prompt(video_path: str, question: str, answer_a: str, answer_b: str) -> str:
    return textwrap.dedent(f"""
    You are a helpful and thoughtful AI assistant with experience in multimodal reasoning.
    ### Task
    Two candidate answers (Model A & Model B) are provided for a question related to a video.
    Your task is to analyze and give a comparative evaluation of their quality and accuracy based on FIVE key dimensions.

    **Evaluation Dimensions**
    1. Fluency and Coherence 
    2. Relevance to the Question and Video 
    3. Accuracy and Completeness 
    4. Reasoning Quality 
    5. Safety and Ethical Alignment 

    **Scoring Guidelines**
    - 9-10: Excellent in all dimensions
    - 6-8: Good overall with minor issues in 1-2 dimensions
    - 3-5: Deficient in 2-3 dimensions
    - 0-2: Poor in 4-5 dimensions

    **Evaluation Process**
    1. First, imagine the most ideal and factually accurate answer to the question based on the video and question context. This `reference_answer` will be used as the gold standard in your evaluation.
    2. Evaluate both answers across all five dimensions.
    3. Assign each model an integer score from 0 to 10 based on the dimensional analysis.
    4. Determine which model performed better overall ("A", "B", or "equal").
    5. Provide detailed reasoning covering all five dimensions.

    **Output Instructions**
    - Your output must be a **strictly valid JSON object**.
    - **Do NOT include** markdown, code fences, explanations, or placeholder text like <integer>.
    - All field names and string values must be enclosed in **double quotes**.
    - Make sure the reasoning is enclosed in a single string under the "reasoning" key.
    - The final verdict should match the better model inside: "<answer>[[A]]</answer>", "<answer>[[B]]</answer>", or "<answer>[[equal]]</answer>".

    ### Required Output Keys
    {{
      "score_A": [integer between 0 and 10],
      "score_B": [integer between 0 and 10],
      "better": "A" or "B" or "equal",
      "reasoning": "<think>Part 1: In terms of Fluency and Coherence, …  
       For Relevance to the Question and Video, …  
       Regarding Accuracy and Completeness, …  
       In terms of Reasoning Quality, …  
       Part 2: In terms of Safety and Ethical Alignment, …</think>",
      "final_verdict": "<answer>[[A]]</answer>"
    }}

    ### Context
    Video file: {video_path}  
    Question: {question}  
    Candidate A: {answer_a}  
    Candidate B: {answer_b}
    """).strip()
