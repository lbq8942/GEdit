# from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from evals.glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
from evals.glue_eval.batched_classification import evaluate_batched
import time
import torch
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 100
## IMPORTANT, few shot learning is important as it allow the answer coming out from the model to be formatted. 

class NLIEval():
    def __init__(self, cfg,model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {MAX_NUMBER_OF_FEW_SHOTS}"
        self.cfg=cfg
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('evals/glue_eval/dataset/nli.pkl', number_of_few_shots, number_of_tests)
        self._initialize_prompts()
        
    # def _initialize_prompts(self):
    #     self.postfix_prompt = 'True or False? answer:' 
    #     self.few_shot_context = ""
    #     for _, few_shot in enumerate(self.few_shots):
    #         self.few_shot_context += f'{few_shot["sentence1"]} entails the {few_shot["sentence2"]} {self.postfix_prompt} {("True" if few_shot["label"] == "entailment" else "False")}\n' 

    def _initialize_prompts(self):
        self.postfix_prompt = 'True or False? Answer:' 
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f'{few_shot["sentence1"]} entails the {few_shot["sentence2"]} {self.postfix_prompt} {("True" if few_shot["label"] == "entailment" else "False")}\n')  

    def _create_prompt(self, example, gen_len):
        question = f'{example["sentence1"]} entails the {example["sentence2"]} {self.postfix_prompt}'
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
        return input_prompt, example['sentence1'], example['sentence2'], self._get_label(example['label'])
    
    def _get_answer(self, generated_text):
        answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()

        if 'true' in answer_text.lower():
            return 1
        elif 'false' in answer_text.lower():
            return 0
        return -1

    def _get_label(self, example_label):
        if 'entailment' == example_label:
            return 1
        return 0

    def evaluate(self, gen_len=3, print_logs=False):
        return evaluate_batched(
            self, gen_len, ["True", "False"], "nli", print_logs=print_logs
        )

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = '/data/akshat/lingua-models/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    nli_eval = NLIEval(model, tokenizer)
    nli_eval.evaluate(print_logs='True')