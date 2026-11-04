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

class MMLUEval():
    def __init__(self, cfg,model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.cfg=cfg
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('evals/glue_eval/dataset/mmlu.pkl', number_of_few_shots, number_of_tests)
        self._initialize_prompts()


    def _initialize_prompts(self):
        self.glue_prompt = "Question: "
        self.postfix_prompt = 'True or False? answer: '
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            prompta = '(A) ' + few_shot['choices'][0] + '\n'
            promptb = '(B) ' + few_shot['choices'][1] + '\n'
            promptc = '(C) ' + few_shot['choices'][2] + '\n'
            promptd = '(D) ' + few_shot['choices'][3] + '\n'
            self.few_shot_context.append(f"Question: {few_shot['question']}\n{prompta + promptb + promptc + promptd}Answer: {self._get_label(few_shot['answer'])}\n")

    def _get_label(self, example_label):
        if example_label == 0:
            return 'A'
        if example_label == 1:
            return 'B'
        if example_label == 2:
            return 'C'
        if example_label == 3:
            return 'D'
    
    def _create_prompt(self, example, gen_len):
        prompt = 'Question: ' + example['question'] + '\n'
        prompta = '(A) ' + example['choices'][0] + '\n'
        promptb = '(B) ' + example['choices'][1] + '\n'
        promptc = '(C) ' + example['choices'][2] + '\n'
        promptd = '(D) ' + example['choices'][3] + '\n'
        question = prompt + prompta + promptb + promptc + promptd + 'Answer:'
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
        return input_prompt, example['question'], example['answer']
    
    # def _create_prompt(self, example):
    #     prompt = 'Question: ' + example['question'] + '\n'
    #     prompta = '(A) ' + example['choices'][0] + '\n'
    #     promptb = '(B) ' + example['choices'][1] + '\n'
    #     promptc = '(C) ' + example['choices'][2] + '\n'
    #     promptd = '(D) ' + example['choices'][3] + '\n'

    #     input_prompt = self.few_shot_context + prompt + prompta + promptb + promptc + promptd + 'Answer:'
    #     return input_prompt, example['question'], example['answer']

    def _get_answer(self, text):
        if 'a\n' in text.lower():
            return 0
        elif 'b\n' in text.lower():
            return 1
        elif 'c\n' in text.lower():
            return 2
        elif 'd\n' in text.lower():
            return 3
        return -1

    def evaluate(self, gen_len=10, print_logs=False):
        return evaluate_batched(
            self, gen_len, ["A", "B", "C", "D"], "mmlu", print_logs=print_logs
        )

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = "/data/akshat/models/gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    mmlu_eval = MMLUEval(model, tokenizer)
    result_dict, stored_generations = mmlu_eval.evaluate(print_logs='True')