# from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from evals.glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
from evals.glue_eval.batched_classification import evaluate_batched
import math
import torch
import time
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 100

class COLAEval():
    def __init__(self, cfg,model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.cfg=cfg
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('evals/glue_eval/dataset/cola.pkl', number_of_few_shots, number_of_tests)
        self._initialize_prompts()


    def _initialize_prompts(self):
        self.prefix_prompt = 'Is this sentence linguistically acceptable?\n'

        self.postfix_prompt = 'Answer: '
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"{self.prefix_prompt}Sentence: {few_shot['sentence']}\nAnswer: {'No' if few_shot['label'] == 0 else 'Yes'}\n")

    
    # def _create_prompt(self, example):
    #     prompt = 'Sentence: ' + example['sentence'] + '\n'

    #     input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

    #     return input_prompt, example['sentence'], example['label']

    def _create_prompt(self, example, gen_len):
        prompt = 'Sentence: ' + example['sentence'] + '\n'
        question = self.prefix_prompt + prompt + self.postfix_prompt
        question_token_length = len(self.tokenizer(question)["input_ids"])
        remaining_token_length = MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP[self.cfg.llms.name] - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 0:
                break 
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        # print(type(example['label']))
        return input_prompt, example['sentence'], example['label']


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('Answer:')[-1].strip().strip()

        if 'yes' in answer_text.lower():
            return 1
        elif 'no' in answer_text.lower():
            return 0

        return -1

    def evaluate(self, gen_len=10, print_logs=False):
        return evaluate_batched(
            self, gen_len, ["Yes", "No"], "cola", print_logs=print_logs
        )

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = "/data/akshat/models/gpt2-xl"
    #model_name = 'EleutherAI/gpt-j-6b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    cola_eval = COLAEval(model, tokenizer)
    result_dict, stored_generations = cola_eval.evaluate(print_logs='True')