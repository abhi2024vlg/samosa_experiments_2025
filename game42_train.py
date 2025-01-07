import os
import pandas as pd
import numpy as np
import time
import random
import sys
import torch
from torch.utils.data import DataLoader,TensorDataset,random_split
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule, LightningModule
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer
import heapq
import pickle
import gzip
import editdistance
import backoff
import openai
import re
import torch.nn.functional as F 
import yaml
import json
from tarski.io import PDDLReader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
device = "cuda" if torch.cuda.is_available() else "cpu"

from contextlib import redirect_stdout, redirect_stderr

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 12

seed_everything(seed)


def load_model(
    pretrained_model,
    device,
    checkpoint_path=None,
    use_4bit=False,
    use_lora=False,
    mode="train"  # Available parameter: "train" or "test"
):
    """
    Load a model with specified parameters.
    
    Args:
        pretrained_model (str): Path or name of the pretrained model
        device (str): Device to load the model on
        checkpoint_path (str, optional): Path to checkpoint - used for both training and testing
        use_4bit (bool, optional): Whether to use 4-bit quantization
        use_lora (bool, optional): Whether to use LoRA
        mode (str, optional): Either "train" or "test"
    
    Returns:
        tuple: (model, tokenizer)
    """
    if checkpoint_path and mode == "train":
        # Loading existing PEFT model for continued training
        model = PeftModel.from_pretrained(
            pretrained_model,
            checkpoint_path,
            is_trainable=True
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    else:
        # Load base model
        ## Currently we are not using 4bit configuration as there is some issue with BitsandBytes dependency
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config
            )
            model = prepare_model_for_kbit_training(model)
            model.config.use_cache = False
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            model.to(device)
            
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            add_bos_token=False
        )
        
        # Apply LoRA or load checkpoint based on mode
        if use_lora and mode == "train":
            lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            )
            
            # Apply LoRA to the model
            model = get_peft_model(model, lora_config)
        
        elif checkpoint_path and mode == "test":
            model = PeftModel.from_pretrained(model, checkpoint_path)

    return model, tokenizer

# For training with LoRA
model, tokenizer = load_model(
    pretrained_model="meta-llama/Llama-3.2-3B",
    device=device,
    checkpoint_path=None,
    use_4bit=False,#BitsandBytes error
    use_lora=True,
    mode="train"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(self, buffer_size, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.reset()

    def reset(self):
        self._buffer = {}

    def add(self, problem, plan, sample, log_reward):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if the plans have already existed in the problem
        if problem not in self._buffer:
            self._buffer[problem] = {
                "sentences": [],
                "exists": set(),
            }
        if plan in self._buffer[problem]["exists"]:
            return
        
        for buffer_item in self._buffer[problem]["sentences"]:
           
            if buffer_item[0] >= log_reward:
                return
            else:
                self._buffer[problem]["exists"].remove(buffer_item[1])
                self._buffer[problem]["sentences"].remove(buffer_item)
                heapq.heapify(self._buffer[problem]["sentences"])
                self._buffer[problem]["exists"].add(plan)
                heapq.heappush(
                    self._buffer[problem]["sentences"],
                    (
                        log_reward,
                        plan,
                        sample
                    ),
                )
                return
                
        self._buffer[problem]["exists"].add(plan)
        if len(self._buffer[problem]["sentences"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[problem]["sentences"],
                (
                           log_reward,
                            plan,
                            sample
                ),
            )
            self._buffer[problem]["exists"].remove(popped[1])
        else:
            heapq.heappush(
                self._buffer[problem]["sentences"],
                (
                    log_reward,
                    plan,
                    sample
                ),
            )

    

    def sample(self, batch_size, problem):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        if problem not in self._buffer:
            return None, None
        prompt_buffer = self._buffer[problem]["sentences"]
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        return [prompt_buffer[i][0] for i in idx], [prompt_buffer[i][1] for i in idx], [prompt_buffer[i][2] for i in idx],
        

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)
            

# 5-shot (NOT in used anywhere right now )
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 42.
Input: 4 7 6 3
Answer: (4 + 3) * (7 - 1) = 42
Input: 2 9 10 21
Answer: 2 * 21 * (10 - 9) = 42
Input: 3 10 3 3
Answer: (3 + 3) * (10 - 3) = 42
Input: 0 4 8 21
Answer: (8 / 4 + 0) * 21 = 42
Input: 13 13 13 3
Answer: 13 + 13 + 13 + 9 = 42
Input: {input}
'''

# 5-shot
cot_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 42. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.  
Input: 4 7 6 3  
Steps:  
4 + 3 = 7 (left: 7 6 7)  
7 - 6 = 1 (left: 7 1)  
7 * 6 = 42 (left: 42)  
Input: 2 9 10 21  
Steps:  
2 * 21 = 42 (left: 9 10)  
10 - 9 = 1 (left: 1 42)  
1 * 42 = 42 (left: 42)  
Input: 3 10 3 3  
Steps:  
3 + 3 = 6 (left: 10 6 3)  
10 - 6 = 4 (left: 4 3)  
4 * 3 = 12 (left: 12 3)  
12 * 3 = 42 (left: 42)  
Input: 0 4 8 21  
Steps:  
8 / 4 = 2 (left: 21 2 0)  
2 + 0 = 2 (left: 21 2)  
21 * 2 = 42 (left: 42)  
Input: 13 13 13 3  
Steps:  
13 + 13 = 26 (left: 26 13 3)  
26 + 13 = 39 (left: 39 3)  
39 + 3 = 42 (left: 42)  
Input: {input}  
Steps:  
'''

# 1-shot
propose_prompt = '''Input: 4 7 6 3  
Possible next steps:  
4 + 7 = 11 (left: 11 6 3)  
4 * 6 = 24 (left: 7 24 3)  
6 / 3 = 2 (left: 4 7 2)  
4 + 3 = 7 (left: 7 6 7)  
7 - 6 = 1 (left: 4 3 1)  
Input: {input}
Possible next steps:
'''

value_prompt = '''Evaluate if given numbers can reach 42 (sure/likely/impossible)
21 21
21 + 21 = 42
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 3 6
4 + 3 + 6 = 7 + 6 = 13
4 * 3 - 6 = 12 - 6 = 6
(4+3) * 6 = 7 * 6 = 42
sure
10 10 20
10 + 20 + 10 = 30 + 10 = 40
I cannot obtain 42 now, but numbers are within a reasonable range
likely
5 4 1
5 + 4 + 1 = 10
(5 - 4) * 1 = 1
5 4 1 are all too small
impossible
100 200 300
100 + 200 + 300 = 600
(100 + 300) * 200 = 80000
100 200 300 are all too big
impossible
{input}
'''

value_last_step_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 42. Given an input and an answer, give a judgment (sure/impossible) if the answer is correct, i.e., it uses each input exactly once and no other numbers, and reaches 42.  
Input: 4 7 6 3  
Answer: (4 + 3) * (7 - 1) = 42  
Judge:  
sure  
Input: 2 9 10 21  
Answer: 2 * 21 * (10 - 9) = 42  
Judge:  
sure  
Input: 3 10 3 3  
Answer: (3 + 3) * (10 - 3) = 42  
Judge:  
sure  
Input: 0 4 8 21  
Answer: (8 / 4 + 0) * 21 = 42  
Judge:  
sure  
Input: 13 13 13 3  
Answer: 13 + 13 + 13 + 9 = 42  
Judge:  
sure  
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible
Input: {input}
Answer: {answer}
Judge:'''


def cot_prompt_wrap(x: str, y:str='') -> str:
    """
    Wraps input text in a chain-of-thought prompt template.

    Args:
        x (str): The primary input text to be wrapped in the prompt template
        y (str): Optional additional text to append to the formatted prompt (default: '')

    Returns:
        str: The complete formatted prompt with the input text and any additional content
    
    Example:
        >>> cot_prompt_wrap("5 2 8 1", "Previous steps...")
        # Returns the formatted prompt with the numbers and steps
    """
    
    return cot_prompt.format(input=x) + y
    
def get_current_numbers(y: str) -> str:
    """
    Extracts the current available numbers from the last line of a solution step.

    Args:
        y (str): A multi-line string containing solution steps, where the last line
                contains remaining numbers in the format "... (left: n1 n2 n3)"

    Returns:
        str: Space-separated string of remaining numbers

    Example:
        >>> get_current_numbers("Step 1: 5 + 3 = 8 (left: 8 2 1)")
        "8 2 1"
    """
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]
    
def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
    
    """
    Calculates a confidence score based on the solution steps and predefined value mappings.

    Args:
        x (str): Input puzzle string
        y (str): Generated solution steps
        value_outputs (list): List of possible value classifications

    Returns:
        float: Confidence score based on the solution quality:
            - 0 for incomplete solutions (less than 4 lines without an answer)
            - 0.001 for impossible solutions
            - 1 for likely solutions
            - 20 for sure solutions
            - Sum of values for multiple classifications

    Example:
        >>> value_outputs_unwrap("5 2 8 1", "Step 1...\nStep 2...", ["likely", "sure"])
        21.0
    """
    
    if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
        return 0
    
    value_names = [_.split('\n')[-1] for _ in value_outputs]
    value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
    value = sum(value * value_names.count(name) for name, value in value_map.items())
    return value

    
def value_prompt_wrap(x: str, y: str) -> str:

    """
    Creates a validation prompt based on the current state of the solution.

    Args:
        x (str): Original input puzzle
        y (str): Current solution step or final answer

    Returns:
        str: A formatted prompt either for:
            - Validating the final answer if the solution is complete
            - Validating the current step based on remaining numbers

    Example:
        >>> value_prompt_wrap("5 2 8 1", "Step 1: 5 + 3 = 8 (left: 8 2 1)")
        # Returns prompt with "8 2 1" for validation
    """
    
    last_line = y
    
    if 'left: ' not in last_line:  # last step
        ans = last_line.lower().replace('answer: ', '')
        return value_last_step_prompt.format(input=x, answer=ans)
    current_numbers = get_current_numbers(y)
    return value_prompt.format(input=current_numbers)

def calculate_and_complete_expression(expression_str,numbers_list):

    """
    Validates and completes a mathematical expression using the available numbers.

    Args:
        expression_str (str): Mathematical expression in the format "num1 op num2 = "
        numbers_list (list): List of available numbers that can be used

    Returns:
        tuple: (
            bool: True if expression is valid and calculable, False otherwise,
            str: Complete expression with result and remaining numbers if valid, None otherwise,
            list: Updated list of available numbers if valid, None otherwise
        )

    Example:
        >>> calculate_and_complete_expression("5 + 3 = ", ["5", "3", "8", "1"])
        (True, "5 + 3 = 8 (left: 8 1)", ["8", "1"])
    """

    try:
        if '=' not in expression_str:
            return False, None, None
        extracted_numbers = re.findall(r'\b\d+\b', expression_str.split('=')[0])
        if len(extracted_numbers) != 2:
            return False, None, None
        num_l = numbers_list[:]
        for num in extracted_numbers:
            if num in num_l:
                num_l.remove(num)
            else: return False, None, None
        left = expression_str.split('=')[0].strip()
        left = left.replace(' ','')
        l = re.split('\+|-|\*|/',left)[0]
        r = re.split('\+|-|\*|/',left)[1]
        op = left[len(l)]
        left = l + ' ' + op + ' ' + r
        math_expression = expression_str.split('=')[0].strip()

        # Use the eval function to calculate the result of the expression
        
        result = eval(math_expression)
        if (result % 1 != 0): return False, None, None
        else: result = int(result)
        num_l.append(str(result))
        lf = ' '.join(num_l)
        complete_expression = left + ' = ' + str(result) + ' (left: ' + lf + ')'

        return True, complete_expression, num_l
    except: return False, None,None
        
def generate_op(num_list):
    
    """
    Generates a random valid mathematical operation using available numbers.

    Args:
        num_list (list): List of available numbers to use in the operation

    Returns:
        tuple: (
            str: Generated mathematical expression with result and remaining numbers,
            list: Updated list of available numbers
        )

    Example:
        >>> generate_op(["5", "3", "8", "1"])
        ("5 + 3 = 8 (left: 8 1)", ["8", "1"])

    Note:
        - Only generates operations that result in positive integers
        - Uses random selection of operators (+, -, *, /) and numbers
        - Continues trying until a valid operation is found
    """
    
    ops = ['+','-','*','/']
    ans = 0.1
    print("Code Generate")
    # print(num_list)
    num_l = num_list[:]
    while(ans < 0 or not isinstance(ans,int)):
        op = random.choice(ops)
        nums = random.sample(num_l,2)
        try:
            ans = eval(nums[0]+op+nums[1])
        except:
            continue
    num_l.remove(nums[0])
    num_l.remove(nums[1])
    num_l.append(str(ans))
    lf = ' '.join(num_l)
    ans_str = nums[0] + ' ' + op + ' ' + nums[1] + ' = ' + str(ans) + ' (left: ' + lf + ')'
    return ans_str, num_l

class InputExample:
    def __init__(self, input_ids, labels, attention_masks, reward):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_masks = attention_masks
        self.reward = reward

class Game24DataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            train_data_path: str,
            batch_size: int = 4,
            device: str = "cuda",
            limit_prompts: int = None,
            do_sft: bool = False,
            max_length: int = 1024,
    ):
        """
        
        Args:
            tokenizer: Tokenizer for processing text
            train_data_path: Path to training data file
            batch_size: Batch size for training
            device: Device to use for computation
            limit_prompts: Limit on number of prompts to process
            do_sft: Whether to do SFT filtering
            max_length: Maximum sequence length
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.train_data_path = train_data_path
        self.batch_size = batch_size
        self.device = device
        self.limit_prompts = limit_prompts
        self.do_sft = do_sft
        self.max_length = max_length
        
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        print(stage)
        if stage == "fit" or stage is None:
            print("Loading data")
            game24full = []
            with open(self.train_data_path, 'r') as f:
                for line in f:
                    game24full.append(json.loads(line))
            
            features = self.convert_train_to_features(game24full)
            all_inputs_ids = torch.stack([f.input_ids for f in features])
            all_labels = torch.stack([f.labels for f in features])
            all_attention_masks = torch.stack([f.attention_masks for f in features])
            all_rewards = torch.Tensor([f.reward for f in features])
            train_data = TensorDataset(all_inputs_ids, all_labels, all_attention_masks, all_rewards)
            self.train_data = PromptDataPipe(train_data)
            
            tests_all = list(pd.read_csv('game42_data/42.csv')['Puzzles'])
            vald = tests_all[10:20] + tests_all[-20:-10]
            self.val_data = PromptDataPipe(vald)
            
        elif stage == 'test':
            tests_all = list(pd.read_csv('game42_data/42.csv')['Puzzles'])
            self.test_data = PromptDataPipe(tests_all[1000:1010])

    def creat_labels(self, inputs, generate_text, ignore_token_id):
        labels = inputs["input_ids"].clone()
        labels[:, :len(inputs["input_ids"][0])-len(self.tokenizer(generate_text)["input_ids"])] = ignore_token_id
        return labels

    def convert_train_to_features(self, examples):
        ignore_token_id = LabelSmoother.ignore_index
        features = []
        i = 0
        query = ''
        
        for example in examples:
            if self.do_sft:
                if example['reward'] != 100:
                    continue
                elif example['idx'] != query:
                    query = example['idx']
                else:
                    continue

            input_prompt = cot_prompt_wrap(example['input']) + example['generate_data']
            generate_text = "Steps: \n" + example['generate_data']
            inputs = self.tokenizer(input_prompt, return_tensors="pt")
            
            if self.max_length < len(inputs["input_ids"][0]):
                print("Input length is greater than max_length")
                print(inputs["input_ids"].shape[1])
                input()
                
            padding_length = self.max_length - inputs["input_ids"].shape[1]
            labels = self.creat_labels(inputs, generate_text, ignore_token_id)
            attention_mask = torch.ones_like(inputs["input_ids"])
            
            padded_input_ids = torch.cat([
                inputs['input_ids'], 
                torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long).unsqueeze(0)
            ], dim=-1)[0]
            
            padded_attention_mask = torch.cat([
                attention_mask, 
                torch.zeros(padding_length, dtype=torch.long).unsqueeze(0)
            ], dim=-1)[0]
            
            padded_labels = torch.cat([
                labels, 
                torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long).unsqueeze(0)
            ], dim=-1)[0]

            features.append(InputExample(
                input_ids=padded_input_ids,
                labels=padded_labels,
                attention_masks=padded_attention_mask,
                reward=example['reward']
            ))

            if i < 5:
                print("***Example: ***", i)
                print("length of input_ids:", len(inputs["input_ids"][0]))
                print("padded_input_tokens:", self.tokenizer.decode(padded_input_ids))
                i += 1
                
        return features

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=2)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=2)

class PromptDataPipe(MapDataPipe):
    def __init__(self, problems) -> None:
        super().__init__()
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):
        return self.problems[index]
    
def lora_to_base(model):

    """
    Disables LoRA adapter layers and sets model to evaluation mode.

    Args:
        model: The neural network model with LoRA adapters

    Note:
        Catches and handles cases where the model doesn't have adapter layers
    """
    
    try:
        model.base_model.disable_adapter_layers()
    except:
        print("No adapter layers to disable")
    model.eval()
    
def base_to_lora(model):
    
    """
    Enables LoRA adapter layers and sets model to training mode.

    Args:
        model: The neural network model with LoRA adapters

    Note:
        Catches and handles cases where the model doesn't have adapter layers
    """
    
    try:
        model.base_model.enable_adapter_layers()
    except:
        print("No adapter layers to enable")
    model.train()

def tb_loss(log_pf, log_r, logz):
    """
    Calculates the trajectory balance loss.

    Args:
        log_pf (tensor): Log probability of forward predictions
        log_r (tensor): Log of rewards
        logz (tensor): Log partition function

    Returns:
        tensor: Mean squared error of the trajectory balance equation

    Note:
        Implements the loss function: L = (log_pf + logz - log_r)²
    """
    loss = (log_pf + logz - log_r) ** 2
    return loss.mean()

class Game24GTNTask(LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        replay_buffer,
        train_data=None,
        val_data=None,
        lr: float = 1e-5,
        logZ_lr: float = 1e-5,
        batch_size: int = 2,
        use_4bit: bool = False,
        do_sft: bool = False,
        test_sample_nums: int = 20
    ):
        """
        PyTorch Lightning module for training a model to solve the Game of 24 using Generative Teaching Networks (GTN).
        
        The Game of 24 is a mathematical game where players must use four numbers and basic arithmetic operations 
        to arrive at the number 24. This implementation uses a language model to generate solution steps and can 
        be trained using either direct SFT (Supervised Fine-Tuning) or GTN approaches.
        
        The module supports:
        - Experience replay for training
        - Value caching for efficient evaluation
        - Multiple training modes (SFT and GTN)
        - 4-bit quantization for memory efficiency
        - Validation and testing of solution generation
        
        Attributes:
        model: The underlying language model
        tokenizer: Tokenizer for processing text inputs
        replay_buffer: Buffer for storing and sampling experiences
        train_data: Training dataset
        val_data: Validation dataset
        lr (float): Learning rate for model parameters
        logZ_lr (float): Learning rate for the logZ parameter in GTN
        batch_size (int): Number of samples per training batch
        use_4bit (bool): Whether to use 4-bit quantization
        do_sft (bool): Whether to use supervised fine-tuning
        test_sample_nums (int): Number of samples to generate during testing
        logZ (Parameter): Learnable parameter for GTN
        value_cache (dict): Cache for storing computed values
        n_samples (int): Number of samples to generate
        test_set (set): Set of unique solutions found during testing
        """
        
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        
        self.model = model
        self.tokenizer = tokenizer
        self.replay_buffer = replay_buffer
        self.train_data = train_data
        self.val_data = val_data
        
        self.lr = lr
        self.logZ_lr = logZ_lr
        self.batch_size = batch_size
        self.use_4bit = use_4bit
        self.do_sft = do_sft
        self.test_sample_nums = test_sample_nums
        
        self.logZ = torch.nn.Parameter(torch.tensor([0], dtype=torch.float))
        self.reward = None
        self.value_cache = {}
        self.n_samples = 10
        self.test_set = set()
        self.get_lr_at_step = lambda step: min(step / 20 * self.lr, self.lr)
        self.ignore_token_id = LabelSmoother.ignore_index

    def get_ll_batch(self, inputs, labels, attention_masks):

        """
        Computes log-likelihoods for a batch of solutions.
        
        Args:
        inputs (tensor): Input token IDs
        labels (tensor): Label token IDs
        attention_masks (tensor): Attention masks
        
        Returns:
        tensor: Batch of log-likelihood scores
        """
        
        with torch.no_grad():
            lora_to_base(self.model)
            outputs = self.model(inputs, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            base_to_lora(self.model)
            return torch.exp(-loss)**(1/0.7)

    def get_ll(self, query, ys):
        
        """
        Computes the log-likelihood of a solution.
        
        Args:
        query (str): The initial problem
        ys (str): The complete solution steps
        
        Returns:
        tuple: (
            float: Computed log-likelihood score,
            dict: Input tensors,
            tensor: Label tensors
        )
        """

        lora_to_base(self.model)
        ignor_token_ids = LabelSmoother.ignore_index

        # Prepare input
        input_prompt = cot_prompt_wrap(query, ys)
        inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)

        # Create labels
        labels = inputs["input_ids"].clone()
        generate_text = "Input: " + query + '\n' + "Steps: \n" + ys

        # Mask unused tokens
        labels[:, :len(inputs["input_ids"][0])-len(self.tokenizer(generate_text)["input_ids"])] = self.ignore_token_id
        
        # Get model outputs
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        base_to_lora(self.model)

        # Convert to likelihood score
        return torch.exp(-loss)**(1/0.7), inputs, labels

    def batch_preprocess(self, preprocessed_samples):
        max_length = max(sample['input_ids'].shape[-1] for sample in preprocessed_samples)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for sample in preprocessed_samples:
            padding_length = max_length - sample['input_ids'].shape[-1]
            
            padded_input_ids = torch.cat([
                sample['input_ids'], 
                torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long).unsqueeze(0)
            ], dim=-1)
            
            padded_attention_mask = torch.cat([
                sample['attention_mask'], 
                torch.zeros(padding_length, dtype=torch.long).unsqueeze(0)
            ], dim=-1)
            
            padded_labels = torch.cat([
                sample['labels'], 
                torch.full((padding_length,), self.ignore_token_id, dtype=torch.long).unsqueeze(0)
            ], dim=-1)
            
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(padded_labels)
        
        return {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "labels": torch.cat(batch_labels, dim=0),
        }

    def query_LM(self, prompt, eos_token_id, do_sample=True, temperature=0.7):

        """
        Queries the language model to generate the next step in the solution.
        
        Args:
        prompt (str): Input prompt for the model
        eos_token_id: End of sequence token ID
        do_sample (bool): Whether to use sampling for generation
        temperature (float): Sampling temperature (higher = more random)
        
        Returns:
        str: First line of generated text or None if generation failed
        
        Notes:
        - Uses top-k sampling with k=10
        - Limits generation to 20 new tokens
        - Returns only the first line of generated text
        """
        
        temperature = temperature if do_sample else 0 ## By default not greedy decoding but sampling based decoding.
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').cuda() ## convert input text to tensor token and pass them to gpu
        attention_mask = torch.ones_like(input_ids)

        
        results = self.model.generate(
            input_ids, 
            do_sample=True, 
            max_new_tokens=20,
            top_k=10, ## Current Sampling strategy top-k . Could change this to top-p, min-p 
            attention_mask=attention_mask,
            use_cache=False,
            temperature=temperature ## For experiments related to temperature ,change value of temperature over here.
        )
        
        results = results[0][len(input_ids[0]):]
        results = self.tokenizer.decode(results, skip_special_tokens=False) ## Converts the generated token IDs back to text, keeping special tokens
        lines = results.splitlines()
        return lines[0] if lines else None

    def get_proposals(self, x, y, nums, do_val=False):
        
        ## Used in game_24_plans function
        
        """
        
        Generates and validates the next step in the Game of 24 solution.
        
        Args:
        x (str): The initial numbers/problem statement
        y (str): Current solution steps
        nums (list): Current available numbers
        do_val (bool): Whether this is being called during validation
        
        Returns:
        tuple: (
            proposals (str): Generated next step in solution,
            numss (list): Updated list of available numbers
        ) or generate_op(nums) if all attempts fail
        
        Notes:
        - Makes up to 3 attempts to generate valid proposals
        - Validates each proposal using calculate_and_complete_expression
        - Falls back to generate_op if all attempts fail
        """

        propose_prompt = cot_prompt_wrap(x, y)
        proposals = self.query_LM(propose_prompt, eos_token_id=self.tokenizer.eos_token_id)
        flag, proposals, numss = calculate_and_complete_expression(proposals, nums)
        if do_val and not flag:
            return None, None
            
        calc = 0
        while calc < 3:
            if flag:
                return proposals, numss
            else:
                flag, proposals, numss = calculate_and_complete_expression(proposals, nums)
                calc += 1
                if flag:
                    return proposals, numss
        return generate_op(nums)

    def get_value(self, x, y, n_evaluate_sample, cache_value=True):
         
        """
         Estimates the value of the current game state using GPT.
         
         Args:
         x (str): Current game state
         y (str): Current solution steps
         n_evaluate_sample (int): Number of samples for evaluation
         cache_value (bool): Whether to use value caching
         
         Returns:
         float: Estimated value of the current state
         
         Notes:
         - Uses value caching for efficiency when enabled
         - Integrates with GPT for state evaluation
         """

        ## It is currently not in use.
        
        value_prompt = value_prompt_wrap(x, y)
        if cache_value and value_prompt in self.value_cache:
            return self.value_cache[value_prompt]
        value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
        value = value_outputs_unwrap(x, y, value_outputs)
        if cache_value:
            self.value_cache[value_prompt] = value
        return value

    def test_output(self, query, output: str):
        
        """
        Tests if a generated solution is valid.
        
        Args:
        query (str): Original problem
        output (str): Generated solution
        
        Returns:
        dict: Contains 'r' key with reward value (0 or 1)
        
        Notes:
        - Verifies that solution uses original numbers
        - Checks if solution evaluates to 24
        - Returns 0 for invalid solutions
        """
        
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', query)
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception:
            return {'r': 0}

    def get_24_plans(self, query, use_gpt_value=True, do_val=False):

        ## It's being only used in validation and test function and at both places we are passing
        ## use_gpt_value=False so gpt is not being used for value estimation.
        
        """
        Generates a complete solution plan for reaching 24 using the given numbers.
        
        Args:
        query (str): Space-separated string of four numbers
        use_gpt_value (bool): Whether to use GPT for value estimation
        do_val (bool): Whether this is being called during validation
        
        Returns:
        tuple: (
            output (str): Complete solution steps,
            sample (dict): Tokenized inputs for model training,
            reward (float): Numerical reward for the solution,
            ll_reward (float): Log-likelihood reward,
            state_nums (list): Final state of numbers
        )
        
        Example:
        Input: "1 2 3 4"
        Output: Complete solution showing steps like:
            "1 + 2 = 3 (left: 3 3 4)
             3 + 3 = 6 (left: 4 6)
             6 * 4 = 24 (left: 24)"
        """
        ys = ''
        sample = None
        ll_reward = None
        infos = []
        values = []
        state_nums = query.split()

        # Generate solution steps
        for step in range(3):
            new_ys, state_nums = self.get_proposals(query, ys, state_nums, do_val)
            if new_ys is None and do_val:
                return "FAIL", None, None, None, [0]
            # Get value estimates if using GPT
            if use_gpt_value:
                values.append(self.get_value(query, new_ys, 1, True))
            infos.append({'step': step, 'x': query, 'ys': ys, 'new_ys': new_ys, 'values': values})
            ys = ys + new_ys + '\n'
            
        output = cot_prompt_wrap(query, ys)
        ll_reward, inputs, labels = self.get_ll(query, ys)
        attention_mask = torch.ones_like(inputs["input_ids"]).to('cpu')
        sample = dict(
            input_ids=inputs["input_ids"].to('cpu'),
            labels=labels.to('cpu'),
            attention_mask=attention_mask
        )
        # Calculate final reward
        # Reward is sum of intermediate values plus large bonus for success
        if use_gpt_value:
            reward = sum(values) + 100*(state_nums[0]=='24')
        else:
            # Binary reward based on success
            if state_nums[0]=='24': reward = 100
            else: reward = 0.001
        output = "Input: " + query + '\n' + "Steps: \n" + ys
        
        return output, sample, reward, ll_reward, state_nums

    def forward_prob(self, input_ids, targets, attention_mask):

        """
        Computes forward probabilities for the model.
        
        Args:
        input_ids (tensor): Input token IDs
        targets (tensor): Target token IDs
        attention_mask (tensor): Attention mask
        
        Returns:
        tensor: Negative log probabilities per sample
        
        Notes:
        - Handles teacher forcing
        - Computes per-token losses
        - Applies attention masking
        """
        
        base_to_lora(self.model)

        # Get model outputs
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift logits and labels for teacher forcing
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        # Calculate per-token losses
        N, L, C = logits.shape

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, C),
            shift_labels.view(-1),
            ignore_index=self.ignore_token_id,
            reduction='none'
        )

        # Sum losses per sample
        loss_per_sample = loss_per_token.view(N, L-1)
        loss_per_sample = loss_per_sample * attention_mask[..., 1:]
        return -loss_per_sample.sum(dim=1)

    def training_step(self, batch, batch_idx):
         
        """
        Performs a single training step.
        
        Args:
        batch (tuple): Batch of training data
        batch_idx (int): Index of current batch
        
        Returns:
        tensor: Computed loss for the batch
        
        Notes:
        - Supports both SFT and GTN training modes
        - In GTN mode, uses Tree-Backup (TB) loss
        - Logs training metrics for monitoring
        """
        
        LOG_PF, LOG_R = [], []
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, labels, attention_mask, rewards = batch

        ## For direct SFT training (Uses direct loss calculation from model outputs)
        if self.do_sft:
            base_to_lora(self.model)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            self.log("train/loss", loss, on_step=True, on_epoch=True,
                    sync_dist=True, prog_bar=True, batch_size=self.batch_size)
            return loss

        ## GTN(Generative Teaching Network) specific training:

        # 1. Get log-likelihood rewards
        ll_rewards = self.get_ll_batch(input_ids, attention_masks=attention_mask, labels=labels)
        # 2. Combine with task rewards and take log
        log_reward_list = torch.log(rewards + ll_rewards)
        # 3. Get forward probabilities
        LOG_PF = self.forward_prob(input_ids, labels, attention_mask)
        # 4. Stack rewards
        LOG_R.extend(log_reward_list)
        LOG_R = torch.stack(LOG_R)

        # 5. Calculate Tree-Backup loss
        loss = tb_loss(log_pf=LOG_PF, log_r=LOG_R, logz=self.logZ)
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                sync_dist=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, problem, batch_idx):
        
        """
        Performs validation on a single problem.
        
        Args:
        problem (list): Problem to validate
        batch_idx (int): Index of validation batch
        
        Notes:
        - Tests solution generation multiple times
        - Logs success rate and other metrics
        - Prints generated solutions for inspection
        """
        
        print("===Validation===")
        test_nums = 5
        success = False
        self.model.eval()
        suc_num = 0
        
        for i in range(test_nums):
            output, sample, reward, ll_reward, sn = self.get_24_plans(problem[0], use_gpt_value=False)
            if sn[0] == '24':
                suc_num += 1
            if not success: success = (sn[0] == '24')
            print(output)
            
        print("SUC_NUM: ", suc_num)
        self.log("val/success", success, on_step=True, on_epoch=True,
                sync_dist=True, prog_bar=True, batch_size=self.batch_size)

    def test_step(self, problem, batch_idx):

        """
        Performs testing on a single problem.
        
        Args:
        problem (list): Problem to test
        batch_idx (int): Index of test batch
        
        Notes:
        - Generates multiple solutions for each problem
        - Tracks unique valid solutions found
        - Logs success rate and solution diversity metrics
        """
        print("===Test===")
        success = False
        self.model.eval()
        self.test_set = set()
        
        for i in range(self.test_sample_nums):
            output, sample, reward, ll_reward, sn = self.get_24_plans(
                problem[0], use_gpt_value=False, do_val=True)
            if sn[0] == '24':
                self.test_set.add(output)
            if not success:
                success = (sn[0] == '24')
            print(output)
            
        print("TRAJ_NUM: ", len(self.test_set))
        self.log("val/success", success, on_step=True, on_epoch=True,
                sync_dist=True, prog_bar=True, batch_size=self.batch_size)
        self.log("val/traj_num", len(self.test_set), on_step=True, on_epoch=True,
                sync_dist=True, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):

        """
        Configures optimizers for training.
        
        Returns:
        optimizer: AdamW optimizer (either 8-bit or standard)
        
        Notes:
        - Supports 4-bit quantization when enabled
        - Uses separate learning rates for model and logZ
        - Returns appropriate optimizer based on configuration
        """
        
        if self.use_4bit: ## Currently 4bit quantisation not supported
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': [self.logZ,], 'lr': self.logZ_lr}
            ])
        else:
            return torch.optim.AdamW([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': [self.logZ,], 'lr': self.logZ_lr}
            ])
            
def game24_planning(
    model,
    tokenizer,
    buffer_size=50, # Default value
    epoch_nums=5,
    do_train=True,
    do_test=False,
    load_checkpoint_path=None,
):
    """
    Plan and execute training/testing for the Game24 task.
    
    Args:
        model: The model to train/test
        tokenizer: Tokenizer for the model
        buffer_size (int): Size of the replay buffer
        epoch_nums (int): Number of training epochs
        do_train (bool): Whether to perform training
        do_test (bool): Whether to perform testing
        load_checkpoint_path (str, optional): Path to checkpoint for testing
    """
    rbuffer = ReplayBuffer(buffer_size=buffer_size)
    
    data = Game24DataModule(
        tokenizer=tokenizer,
        train_data_path="game42_data/train_trimmed.json",
        batch_size=4, ## As per code.
        device="cuda",
        limit_prompts=None,
        do_sft=False, #
        max_length=1024
    )
     
    train_data = list(pd.read_csv('game42_data/42.csv')['Puzzles'])
    logZ = torch.nn.Parameter(torch.tensor([0], dtype=torch.float))
    
    
    task = Game24GTNTask(
        model=model,
        tokenizer=tokenizer,
        replay_buffer=rbuffer,
        train_data=train_data,
        val_data=train_data,
        lr=1e-5,
        logZ_lr=1e-5,
        batch_size=4,
        use_4bit = False,
        do_sft = False,
        test_sample_nums = 20
    )
    # Create a logger
    
    logger = TensorBoardLogger("logs", name="game42_experiment")
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision=16,
        max_epochs=epoch_nums,
        num_sanity_val_steps=0,
        logger=logger
    )
    
    if do_train:
        trainer.fit(model=task, datamodule=data)
        if do_test:
            trainer.test(model=task, datamodule=data)
    elif do_test:
        print("Start Test")
        trainer.test(
            model=task, 
            datamodule=data,
            ckpt_path=load_checkpoint_path
        )
        

with open('training_log.txt', 'a') as f:
    with redirect_stdout(f), redirect_stderr(f):
        game24_planning(
            model=model,
            tokenizer=tokenizer,
            buffer_size=50,
            epoch_nums=5,
            do_train=True,
            do_test=False
        )
