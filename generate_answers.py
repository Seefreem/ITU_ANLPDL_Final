import os
import pickle
import torch
import glob
import numpy as np 

from tqdm import tqdm  # For a progress bar
def generate_answers_and_save(dataset, model, tokenizer, output_dir):
    """
    For each QA pair in the dataset, this function:
      1. Constructs a prompt using the given template.
      2. Uses the model to generate 5 answers (with beam search) while also 
         returning hidden states.
      3. Extracts, for each generated answer, the hidden state vector corresponding 
         to its last token from each layer.
      4. Saves the generated answers and hidden representations to disk.
    
    The results are saved as a pickle file in output_dir.
    """
    print('generate_answers_and_save:')
    # results = []
    
    # Ensure the model is in evaluation mode and on the correct device.
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process each QA pair.
    for idx, ditem in enumerate(tqdm(dataset, desc="Processing QA pairs")):
        # Construct the prompt.
        # (Note: we use ditem['question'] because we have flattened the dataset.)
        prompt = (
            f"Reading the passage and answer given question concisely.\n\n"
            f"Passage:\n{ditem['story']}\n\n"
            f"Question: {ditem['question']}\nAnswer:"
        )
        
        # Tokenize the prompt.
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate 5 candidate answers.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=5,
                num_beams=5,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_hidden_states=True,  # Request hidden states.
            )
        
            # The generated token ids; shape: (5, sequence_length)
            generated_ids = outputs.sequences
            answers = []
            # last_hidden_states_all = []
            # The prompt length so we can extract only the generated part.
            prompt_length = inputs["input_ids"].shape[1]
            
            # For each generated answer:
            for i in range(generated_ids.shape[0]):
                seq = generated_ids[i]
                # Remove the prompt tokens to get the generated answer text.
                answer_ids = seq[prompt_length:]
                answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                answers.append(answer_text)
                # print('answer_text', answer_text)
                
            # Extract the last token representation of the input prompt from each layer.
            # outputs.hidden_states is a tuple of length L (usually L = num_layers + 1,
            # where the first element is the embeddings output).
            last_token_reps = []
            '''
            shape: (step, layers, beams, token, vocab)
            lken(outputs.hidden_states): 20
            len(element): 27
            layer torch.Size([5, 346, 2304])
            layer torch.Size([5, 1, 2304])
            '''
            for layer_hidden in outputs.hidden_states[0]:
                # layer_hidden has shape (num_return_sequences, seq_len, hidden_size)
                # For the current generated sequence, pick the last token.
                last_token_rep = layer_hidden[0, -1, :].detach().cpu().float().numpy().astype(np.float16) 
                last_token_reps.append(last_token_rep)
            # last_hidden_states_all.append(last_token_reps)
            
            # Append results for this QA pair.
            result = {
                "prompt": prompt,
                "generated_answers": answers,  # List of 5 answer strings.
                "last_hidden_states": last_token_reps,  # List of 5 items, each a list of layer vectors.
                "reference answer": ditem['answer']
            }
            # print('Check:', result['prompt'], result['generated_answers'], result['reference answer'])
            # Save the result to a separate file.
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f"result_{idx:06d}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(result, f)


def judge_answers_in_pickles(save_dir, model, tokenizer):
    """
    Loads all pickle files from `save_dir` (which should contain files like "result_*.pkl")
    and, for each file, uses the LLM to judge each generated answer.

    For each file, the following steps are performed:
      1. Remove the prefix "Reading the passage and answer given questions concisely." from the stored prompt.
      2. Prepend the system prompt: 
         "You are an examiner and your job is to assess if an answer is correct, given the context, question and the reference answer."
      3. For each candidate answer, append the candidate answer and the reference answer to the prompt.
      4. Instruct the LLM: "If the answer is correct, respond with \"True\", else respond with \"False\"."
    
    The generated judgment (ideally either "True" or "False") is stored in a new key "judgements" in the pickle file.
    
    The updated pickle files are saved in a subdirectory called "judged_results" (created alongside save_dir).
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Find all pickle files matching "result_*.pkl" inside save_dir.
    pickle_files = glob.glob(os.path.join(save_dir, "result_*.pkl"))
    print('pickle_files:', pickle_files)
    if not pickle_files:
        print(f"No pickle files found in {save_dir}.")
        return

    # Create a directory to store the judged results.
    judged_dir = os.path.join(os.path.dirname(save_dir), "judged_results")
    os.makedirs(judged_dir, exist_ok=True)
    
    # Define the generation parameters for the judging step.
    judge_gen_kwargs = {
        "max_new_tokens": 300,        # short response is expected ("True" or "False")
        "num_return_sequences": 1,
        "do_sample": True,
        'temperature': 0.8
    }
    
    # The prefix that needs to be removed from the stored prompt.
    gen_prefix = "Reading the passage and answer given question concisely."
    
    # Process each pickle file.
    for idx, file_path in enumerate(pickle_files):
        with open(file_path, "rb") as f:
            result = pickle.load(f)
        
        original_prompt = result.get("prompt", "")
        # Remove the generation prefix if it exists.
        if original_prompt.startswith(gen_prefix):
            cleaned_prompt = original_prompt[len(gen_prefix):].strip()
        else:
            cleaned_prompt = original_prompt.strip()
        
        # Prepare to store the LLM's judgments for each generated answer.
        judgements = []
        
        # For each candidate answer, build a judge prompt and get the judgment.
        for candidate_answer in result.get("generated_answers", []):
            # Construct the judge prompt:
            #   1. Add the system instruction.
            #   2. Append the cleaned prompt (which contains the passage and question).
            #   3. Append the candidate answer and the reference answer.
            #   4. Append the instruction for the expected output.
            judge_prompt = (
                "## System prompt\n You are an critical examiner and your job is to assess if an answer is essentially the same as the reference answer.\n\n"
                "Asswss the candedate answer and the reference answer Through the following four criteria: "
                "1) Identifying subtle differences, 2)evaluating clarity and precision, 3) semantc equivalency, and 4)critical thinking. "
                "\nIf they are essentially the same answer to the target question, respond with \"True\", else respond with \"False\"."
                # "\n## Examples:"
                # "\n## Question: Which film was predicted to win?"
                # "\nReference Answer: { Boyhood }"
                # "\nCandidate Answer: { \"Birdman\" }"
                # "\nExplanation: The candidate answer means that Birdman was predicted to win. It means differently from the reference answer.\n"
                # "\n```json\n{\n\"Assessment\": False\n}\n```"

                # "\n## Question: Which film was predicted to win?"
                # "\nReference Answer: { Boyhood }"
                # "\nCandidate Answer: { \"Boyhood\" and \"Birdman\" }"
                # "\n Explanation: The candidate answer means that both Boyhood and Birdman were predicted to win. However in the reference answer, only Boyhood was predicted to win. \n"
                # "\n```json\n{\n\"Assessment\": False\n}\n```"

                # "\n## Question: Which film was predicted to win?"
                # "\nReference Answer: { Boyhood or Birdman }"
                # "\nCandidate Answer: { \"Boyhood\" }"
                # "\n Explanation: The candidate answer means that Boyhood was predicted to win. It complies with the reference answer where Boyhood or Birdman was predicted to win. \n"
                # "\n```json\n{\n\"Assessment\": True\n}\n```"

                

                # "\n## Question: Which film was predicted to win?"
                # "\nReference Answer: { Boyhood and Birdman }"
                # "\nCandidate Answer: { \"Boyhood\" }"
                # "\n Explanation: The candidate answer means that Boyhood was predicted to win. It does not complies with the reference answer where both Boyhood and Birdman were predicted to win. . \n"
                # "\n```json\n{\n\"Assessment\": False\n}\n```"


                # "\n## Question: Which film was predicted to win?"
                # "\nReference Answer: { Boyhood }"
                # "\nCandidate Answer: { \"Boyhood\" was predicted to win. }"
                # "\n Explanation: The candidate answer is a paraphrase of the reference answer. They mean the same thing. \n"
                # "\n```json\n{\n\"Assessment\": True\n}\n```"

                
                # "\n## Question: Which film was predicted to win?"
                # "\nReference Answer: { Boyhood }"
                # "\nCandidate Answer: { \"Boyhood\" and \"Birdman\" were predicted to win. }"
                # "\n Explanation: The candidate answer means that both Boyhood and Birdman were predicted to win. It is not the same as the reference answer where only Boyhood was predicted to win. \n"
                # "\n```json\n{\n\"Assessment\": False\n}\n```"
                
                
                "\n## Task\nNow assess the following candedate answer:"
                f"\n## Context\n{ {cleaned_prompt} }" # cleaned_prompt.find('Question:'):cleaned_prompt.find('Answer:')
                f"\nReference Answer: {{ {result.get('reference answer', '') } }}"
                f"\nCandidate Answer: {{ {candidate_answer} }}"
                # "\nConsidering the examples, and think step-by-step, explain your thoughts, then conclude with JSON format: {Assessment: True or False}"
                "\n Conclude with JSON format: { Assessment: True or False }"
            )
            if idx < 10:
                print('\n\n====judge prompt:', judge_prompt)
            # Tokenize the judge prompt.
            inputs = tokenizer(judge_prompt, return_tensors="pt")
            prompt_length = inputs["input_ids"].shape[1]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate the judgement.
            with torch.no_grad():
                output = model.generate(**inputs, **judge_gen_kwargs)
            
            # Decode the generated text.
            generated_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip().lower()
            if idx < 10:
                print('--->generated_text:', generated_text)
            truncation_idx = generated_text.find('json') if generated_text.find('json') != len(generated_text) else generated_text.find('assessment') 
            generated_text = generated_text[:]
            # Interpret the generated text: if it contains "True" or "False", extract that.
            if "true" in generated_text:
                judgement = "True"
            # elif "False" in generated_text:
            else:
                judgement = "False"
            # else:
            #     judgement = generated_text  # fallback to the raw text
            judgements.append(judgement)
        
        # Add the judgements to the result dictionary.
        result["judgements"] = judgements
        # print('The judgement:', judgements)
        result.pop('last_hidden_states') # to save more space
        
        # Save the updated result into the judged_results directory.
        file_name = os.path.basename(file_path)
        judged_file_path = os.path.join(judged_dir, file_name)
        with open(judged_file_path, "wb") as f:
            pickle.dump(result, f)
        
        if idx < 10:
            print(f"Processed {file_name}: Judgements: {judgements}")
            # break