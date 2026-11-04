# from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from evals.glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
from evals.glue_eval.batched_classification import evaluate_batched
import time
import torch
import numpy as np

MAX_NUMBER_OF_FEW_SHOTS = 100


class SSTEval():
    def __init__(self, cfg,model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.cfg = cfg
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('evals/glue_eval/dataset/sst2.pkl', number_of_few_shots, number_of_tests)
        self._initialize_prompts()


    def _initialize_prompts(self):
        self.prefix_prompt = 'Review :'
        self.postfix_prompt = '\nSentiment :'
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"{self.prefix_prompt} {few_shot['sentence']}{self.postfix_prompt} {'positive' if few_shot['label']==1 else 'negative'}\n")

    def _create_prompt(self, example, gen_len):
        question = self.prefix_prompt + example['sentence'] + self.postfix_prompt
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
        return input_prompt, example['sentence'], example['label']


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('Sentiment :')[-1].strip().strip()

        if 'positive' in answer_text.lower():
            return 1
        elif 'negative' in answer_text.lower():
            return 0

        return -1


    def evaluate(self, gen_len=3, print_logs=False):
        return evaluate_batched(
            self, gen_len, ["positive", "negative"], "sst", print_logs=print_logs
        )

if __name__ == '__main__':
    # Load the tokenizer and model
    sst_eval = SSTEval(None, None)
    exit()
    model_name = 'EleutherAI/gpt-j-6b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    sst_eval = SSTEval(model, tokenizer)
    correct, incorrect, invalid, total = sst_eval.evaluate(print_logs='True')