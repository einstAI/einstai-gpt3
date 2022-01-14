#  Copyright (c) Microsoft Corporation. 
#  Copyright(c) EintAI Inc and Whtcorps Inc
#  Licensed under the MIT license. 
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


END_OF_TURN_TOKEN = '<|endofturn|>'
END_OF_TEXT_TOKEN = '<|endoftext|>'
PROJECT_FOLDER = os.path.dirname(__file__)


def get_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    return tokenizer, model


def prepare_text(text):
    text = text.replace('\n', ' ').replace('\r', '')

    if len(text) > 0 and text[-1] != '.':
        text += '.'

    return text


def generate_response(tokenizer, model, prompt):
    prompt = prepare_text(prompt) + END_OF_TURN_TOKEN + END_OF_TEXT_TOKEN

    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)  # Batch size 1

    output = model(input_ids)[0][0].detach().numpy()  # [CLS] X [SEP] X tokens -> tokens [SEP] [CLS] -> tokens

    generated = tokenizer.decode(output[:, :])  # Remove the end of text token ([SEP]) and the start of the  the text token ([CLS])

    return generated.replace(END_OF_TURN_TOKEN, '').replace(END_OF_TEXT_TOKEN, '')


def generate_response_from_file(tokenizer, model, prompt):
    with open(prompt) as f:
        prompt = prepare_text(f.read()) + END_OF_TURN_TOKEN + END_OF_TEXT_TOKEN

    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)  # Batch size 1

    output = model(input_ids)[0][0].detach().numpy()  # [CLS] X [SEP] X tokens -> tokens [SEP] [CLS] -> tokens

    generated = tokenizer.decode(output[:, :])  # Remove the end of text token ([SEP]) and the start of the text token ([CLS])

    return generated.replace(END_OF_TURN_TOKEN, '').replace(END_OF_TEXT_TOKEN, '')