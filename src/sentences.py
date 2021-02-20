import logging
import numpy as np
import torch
import os

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer)

#logging.basicConfig(
#    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
#)
#logger = logging.getLogger(__name__)

class SentenceGeneration():
    def __init__(self, model_path):
        self.model_name_or_path = os.path.join(os.getcwd()+model_path)
        self.def_length = 100
        self.temperature = 1.0 # default: 1.0 has no effect, lower leads to greedy sampling
        self.repetition_penalty = 1.0 #default: 1.0
        self.k = 50
        self.num_return_sequences = 3
        self.p = 0.9
        self.stop_token = "<EOS>"
        self.device = 'cuda'
        self.n_gpu = 1 
        self.MAX_LENGTH = int(10000)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name_or_path, cls_token="[CLS]", unk_token="[UNK]")
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path)
        self.model.to('cuda')
        self.length = self.adjust_length_to_model(self.def_length, max_sequence_length=self.model.config.max_position_embeddings)
        

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)
    
    def adjust_length_to_model(self, length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = self.MAX_LENGTH  # avoid infinite loop
        return length

    def clean_sentence(self, sent, idx):
        replace_elem = ['\xa0', '<PAD>']
        for elem in replace_elem:
            sent = sent.replace(elem, ' ')
        return sent

    def generate_sentence(self, prompt_text):
        self.set_seed(42)
        # Defining tokenizer and model loading
        #special_tokens = {"cls_token":"[CLS]", "unk_token":"[UNK]"}
         #logger.info(args)
        ### Change default prompt_text
        # Include prompts selection here!
        #prompt_text = prompt if args.prompt else input("Model prompt >>> ")
        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=self.length + len(encoded_prompt[0]),
            temperature=self.temperature,
            top_k=self.k,
            top_p=self.p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            num_return_sequences=self.num_return_sequences,
        )
        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = ""

        for gen_idx, generated_sequence in enumerate(output_sequences):
            #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(self.stop_token) if self.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (prompt_text + text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :])
            total_sequence = self.clean_sentence(total_sequence, gen_idx)
            
            generated_sequences += total_sequence
            
        return generated_sequences
