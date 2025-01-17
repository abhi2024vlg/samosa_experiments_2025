import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
import json
import os
from pytorch_lightning.loggers import TensorBoardLogger
from transformers.trainer_pt_utils import LabelSmoother
from torch.distributions import Categorical
import csv
import pytorch_lightning as pl
from contextlib import redirect_stdout, redirect_stderr
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import heapq
import argparse


#================================================================

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--temp', type=float, default=0.3)
    # parser.add_argument( '--p', type=float, default=0.9)
    parser.add_argument('--finetuned', action='store_true', help="Specify if the model is finetuned")
    return parser.parse_args()

args = options()

print("Temperature is", args.temp, "Finetuned or not", args.finetuned)

nametext = 'finetuned' if args.finetuned else 'pretrained'

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 42

seed_everything(seed)

class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(self, buffer_size, prb=True, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.prb = prb
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
            
        if len(self._buffer[problem]["sentences"]) > self.buffer_size:

            popped = heapq.heappop(self._buffer[problem]["sentences"])
            self._buffer[problem]["exists"].discard(popped[1])

    def sample(self, batch_size, problem):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        if problem not in self._buffer:
            return None, None, None
        prompt_buffer = self._buffer[problem]["sentences"]
        sorted_buffer = sorted(prompt_buffer, key=lambda x: x[0])
        idx_list = np.arange(len(prompt_buffer))
        
        if self.prb:

            
            priorities  = [item[0] for item in prompt_buffer]
            priorities = torch.tensor(priorities, dtype=torch.float32)  
            priorities = priorities - torch.max(priorities)  

            probabilities = torch.exp(priorities) / torch.sum(torch.exp(priorities))

            idx = torch.multinomial(probabilities, batch_size, replacement=True)
        else:
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


pretrained_model = "meta-llama/Llama-3.2-3B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_quant_type="nf4",
    llm_int8_threshold=6.0,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B",
                                            trust_remote_code=True,
                                            device_map="auto",
                                            torch_dtype=torch.bfloat16,
                                            quantization_config=bnb_config)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model,
    add_bos_token=False
)


if args.finetuned:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    memory = torch.load("finetuned-3b/model-epoch39-3b-prontoqa.ckpt")
    def fix_state_dict_keys(checkpoint_state_dict):
        """
        Fixes key mismatches between checkpoint and model state dict by:
        1. Removing the initial 'model.' prefix
        2. Handling quantization-specific keys

        Args:
            checkpoint_state_dict (dict): The loaded checkpoint state dict

        Returns:
            dict: Modified state dict with corrected keys
        """
        new_state_dict = {}

        # Skip special keys
        special_keys = ['logZ']

        for key, value in checkpoint_state_dict.items():
            # Skip special keys
            if key in special_keys:
                continue

            # Skip quantization-specific keys
            if any(x in key for x in ['.absmax', '.quant_map', '.nested_absmax', 
                   '.nested_quant_map', '.quant_state']):
                continue

            # Remove the initial 'model.' prefix
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.'
            else:
                new_key = key

            new_state_dict[new_key] = value

        return new_state_dict
    old_state_dict = memory["state_dict"]
    new_state_dict = fix_state_dict_keys(old_state_dict)
    model.load_state_dict(new_state_dict, strict=True)


    print("Loading finetuned model")


# =========================================================
# ANOTHER WAY OF LOADING WITHOUT 4BIT

# pretrained_model = "meta-llama/Llama-3.2-3B"

# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#      device_map="auto"
# )

# model.to(device)

# tokenizer = AutoTokenizer.from_pretrained(
#     pretrained_model,
#     add_bos_token=False
# )


class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        prompt_file,
        data_file,
        cot_file,
        train_size=20,
        val_size=50,
        test_size=50,
        random_seed=42  # Added for reproducibility
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        
        # Load the base prompt template from JSON file
        with open(prompt_file, 'r') as f:
            self.base_prompt = json.load(f)["input"]
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_seed = random_seed
        
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        self.data_file = data_file
        self.cot_file = cot_file
        
        # Keywords for filtering
        self.ood_keywords = ["3", "31", "131071", "real", "number", "imaginary", "numbers"]
        self.indis_keywords = ["bony", "insect", "cold-blooded", "animal"]
        
        # Add distribution tracking
        self.distribution_stats = {
            'total_in_distribution': 0,
            'total_out_distribution': 0,
            'train_distribution': {},
            'val_distribution': {},
            'test_distribution': {}
        }

    def analyze_distribution(self, data, split_name):
        """
        Analyze the distribution of examples in a dataset split
        """
        in_dist = sum(1 for example in data if any(
            keyword in example[0] for keyword in self.indis_keywords))
        out_dist = sum(1 for example in data if any(
            keyword in example[0] for keyword in self.ood_keywords))
        
        self.distribution_stats[f'{split_name}_distribution'] = {
            'in_distribution': in_dist,
            'out_distribution': out_dist,
            'total': len(data)
        }
        return in_dist, out_dist

    def get_distribution_stats(self):
        """
        Get complete distribution statistics for all splits
        """
        stats = {
            'total_stats': {
                'in_distribution': self.distribution_stats['total_in_distribution'],
                'out_distribution': self.distribution_stats['total_out_distribution']
            },
            'splits': {
                'train': self.distribution_stats['train_distribution'],
                'validation': self.distribution_stats['val_distribution'],
                'test': self.distribution_stats['test_distribution']
            }
        }
        return stats

    def setup(self, stage=None):
        import random
        random.seed(self.random_seed)  # Set seed for reproducibility
        
        in_dist_data = []
        ood_data = []
        potential_train_data = []
        
        # Load and process main dataset
        with open(self.data_file, 'r') as f:
            all_data = json.load(f)

        # Process and categorize all examples
        for key in all_data:
            if key.startswith("example"):
                example = all_data[key]["test_example"]
                actions = example["question"]
                query = example["query"]
                plan = example["chain_of_thought"]
                gt = example["answer"]
                example_data = [actions, query, plan, gt]
                
                if any(keyword in actions for keyword in self.ood_keywords):
                    ood_data.append(example_data)
                elif any(keyword in actions for keyword in self.indis_keywords):
                    in_dist_data.append(example_data)
                    potential_train_data.append(example_data)

        # Store total distribution statistics
        self.distribution_stats['total_in_distribution'] = len(in_dist_data)
        self.distribution_stats['total_out_distribution'] = len(ood_data)

        # Ensure reproducible splitting
        random.shuffle(potential_train_data)
        
        # Create training set
        self.train_data = potential_train_data[:self.train_size]
        remaining_in_dist = potential_train_data[self.train_size:]
        
        # Create validation set
        self.val_data = remaining_in_dist[:self.val_size]
        
        # Create test set (using out-of-distribution data)
        random.shuffle(ood_data)  # Shuffle OOD data
        self.test_data = ood_data[:self.test_size]

        # Analyze distributions for each split
        self.analyze_distribution(self.train_data, 'train')
        self.analyze_distribution(self.val_data, 'val')
        self.analyze_distribution(self.test_data, 'test')

        # Print detailed distribution information
        print("\nDataset Distribution Analysis:")
        print(f"Total in-distribution examples available: {len(in_dist_data)}")
        print(f"Total out-of-distribution examples available: {len(ood_data)}")
        print("\nSplit Sizes:")
        print(f"Training set: {len(self.train_data)} examples")
        print(f"Validation set: {len(self.val_data)} examples")
        print(f"Test set: {len(self.test_data)} examples")

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=1)
        
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1)
            
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1)



def get_full_reward(gt, actions, sum="sum"):
    
    """
    Calculate reward scores for a sequence of actions based on ground truth comparisons.
    
    The function evaluates how well the given actions match with ground truth actions,
    assigning rewards for correct matches in sequence until the first mismatch.
    
    Args:
        gt (list): Ground truth data structure where actions are stored at odd indices
                  Each gt[2i+1][0] contains a ground truth action string
        actions (list): List of predicted actions to evaluate
        sum (str, optional): Method to aggregate rewards. Either "sum" or "avg".
                           Defaults to "sum".
    
    Returns:
        torch.Tensor: Aggregated reward value. Returns 0.0001 if no rewards earned.
                     For sum="sum", returns sum of all rewards.
                     For sum="avg", returns mean of all rewards.
    """
    # Initialize reward tensor on GPU
    reward = torch.zeros(len(actions), dtype=torch.float32, device="cuda:0")

    # Calculate number of possible actions from ground truth structure
    num_feasible_actions = int((len(gt)-1)/2)

    # Track number of correct matches
    j = 0

    # Evaluate each action against ground truth
    for i in range(num_feasible_actions):
        gt_action = gt[2*i+1][0]
        # Check if current action exists and matches ground truth (case-insensitive)
        if (i < len(actions)) and (actions[i].lower() in gt_action.lower()):
            reward[i] += 100 # Assign reward for correct match
            j += 1
        else:
            break # Stop at first mismatch

    # Aggregate rewards based on specified method
    if sum=="sum":
        ret  = torch.sum(reward, dtype=torch.float32) 
    elif sum == "avg":
        ret  = torch.mean(reward, dtype=torch.float32) 

    # Ensure non-zero return value
    if ret.item() == 0:
        ret += 0.0001
    return ret


def eval_tf(last_state, query, answer):
    """
    Evaluate true/false questions by comparing the last state with the query.
    
    This function handles special logic for negations ("not") when evaluating
    true/false answers against the given state.
    
    Args:
        last_state (str): The final state to compare against
        query (str): The true/false question, starting with "True or false: "
        answer (str): The proposed answer ("True" or "False")
    
    Returns:
        bool: True if the answer is correct given the last_state and query
    """
    # Compare normalized sets of words (excluding "not")
    if set(last_state.lower().replace("not", "").split()) == set(query[len("True or false: "):].lower().replace("not", "").split()):
        # finish:
        if answer == "True":
            # For True answers, states must match exactly
            if query[len("True or false: "):].lower() == last_state.lower():
                return True
        else:
            # For False answers, check if "not" appears in exactly one of the statements
            if "not" in set(last_state.lower().split()) - set(query[len("True or false: "):].lower().split()) or "not" in set(query[len("True or false: "):].lower().split()) - set(last_state.lower().split()):
                return True

    return False

def is_finish(last_state, query):
    """
    Determine if the current state represents a finished condition.
    
    Checks various conditions that indicate the reasoning process should terminate.
    
    Args:
        last_state (str): Current state to evaluate
        query (str): The true/false question being processed
    
    Returns:
        bool: True if termination conditions are met, False otherwise
    """
    # Check termination conditions:
    # 1. Contains parentheses
    # 2. Contains "No conclusion" statement
    # 3. First word of query not in last_state
    if ("(" in last_state.strip())  or ("No conclusion can be drawn from these facts".lower() in last_state.lower()) or (query[len("True or false: "):].lower().split()[0] not in last_state.lower()):
        print("FINISH!!!!:\n", last_state, query)
        return True

    # Compare normalized word sets
    return set(last_state.strip().lower().replace("not", "").split()) == set(query[len("True or false: "):].lower().replace("not", "").split())


def query_LM(world_model, world_tokenizer, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.7, max_new_tokens=50):
    temperature = temperature if do_sample else 0
    all_results = []
    input_ids = world_tokenizer.encode(prompt, return_tensors='pt').to(world_model.device)
    
    results = world_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=world_tokenizer.eos_token_id)

    input_ids_list = input_ids.squeeze().tolist()
    
    input_len = len(input_ids_list)

    results = world_tokenizer.decode(results[0][input_len:], skip_special_tokens=False)
    last_newline_position = results.find('\n')

    results = results[:last_newline_position] if last_newline_position != -1 else results
    all_results.append(prompt + results)
    return all_results


# Initialize task with direct parameters
class BlocksWorldGFNTask(LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        replay_buffer,
        train_data=None,
        val_data=None,
        test_data=None,
        use_lora=True
    ):
        
        super().__init__()

        self.sum_avg="sum"
        self.batch_size = 1
        ## Storing 
        self.test_csv = f"test_success, temp={args.temp}, {nametext}.csv"
        # Store model and parameters
        self.model = model
        if use_lora:
            base_to_lora(self.model) # Convert base model to LoRA for efficient fine-tuning
        # Store other components
        self.tokenizer = tokenizer
        self.reward = None
        self.replay_buffer = replay_buffer
        # Store datasets
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.step = 2
        # Load or initialize transition cache
        transition_path = f"transitions/{self.step}/transition.json"
        self.wrong_transitions = {}
        self.ls_wrong_transitions = {}

        if os.path.exists(transition_path):
            with open(transition_path, 'r') as f:
                self.transitions = json.load(f)
        else:
            self.transitions = {}


    @torch.no_grad()
    def validation_step(self, problem, batch_idx):
        # Prepare model for evaluation
        base_to_lora(self.model)   
        self.model.eval()           

        # Unpack problem components
        ACTIONS, QUERY, PLAN, GT = problem
        ACTIONS = ACTIONS[0]
        QUERY = QUERY[0]
        GT = GT[0]

        # Initialize tracking variables
        total_success = 0  # Count of successful solutions
        total_proof_success = 0  # Count of solutions following proof steps
        success_text = []  # Store successful trajectories

        # Try multiple random samples to find different solutions
        for _ in range(20):

            (generated_text,actions,states,reward,sample) = self.generate_trajectories_v2(query = QUERY,allowed_actions = ACTIONS,gt = GT,plan = PLAN,temperature=args.temp,eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],mode="test")

            # Track successful solutions
            if eval_tf(states[-1], QUERY, GT):
                total_success += 1
                actions_joined = '==>'.join(actions)
                states_joined = '==>>'.join(states)
                if actions_joined not in success_text:
                    success_text.append((QUERY,GT ,actions,states))
                # Verify proof steps
                last_3_plans = [PLAN[-6][0], PLAN[-4][0],PLAN[-2][0]]
            
                print("LAST_3__PLANS")
                print(last_3_plans)
                print("-" * 80)
                print("ACTIONS")
                print(actions)
                print("-" * 80)

                if "Finish" not in actions[-1]:
                    last_3_actions = actions[-3:]
                else:
                    last_3_actions = actions[-4:-1]
                
                print("LAST_3__ACTIONS") 
                print(last_3_actions) 
                print("-" * 80)
                
                if last_3_actions == last_3_plans:
                    total_proof_success += 1
                    
                
        # Save successful solutions to CSV
        with open(self.test_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(success_text)

        # Convert counts to binary success indicators
        success = 1 if total_success > 0 else 0
        psuccess = 1 if total_proof_success > 0 and total_success > 0 else 0
    
        # Log metrics
        self.log(
            "test/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size
        )

        self.log(
            "test/psuccess",
            psuccess,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size
        )

    def generate_trajectories_v2(self,
                            query,
                            allowed_actions,
                            gt,
                            plan,
                            temperature,
                            eos_token_id, 
                            max_steps=10,
                            argmax=False,
                            mode="train",
                          ):
        
        # Preprocess allowed actions by splitting and formatting
        allowed_actions = allowed_actions.split(". ")
        print("allowed_actions:")
        print(allowed_actions)
        initial_state = allowed_actions[-1] # Last element is initial state
        allowed_actions = [a+"." for a in allowed_actions[:-1]] # Add periods back to actions
        print("query:")
        print(query)
        print("allowed_actions:")
        print(allowed_actions)
        print("gt:")
        print(gt)
        print("plan:\n", plan)

        # Initialize trajectory tracking variables
        last_state = initial_state
        print("last_state:\n", last_state)
        
        actions = []
        finish = False
        step = 0
        states = []

        # Main trajectory generation loop
        while not finish and (step <= max(len(plan)+1, max_steps)) and len(allowed_actions) > 0:
            base_to_lora(self.model) # Activate LoRA adaptations
            self.model.eval() # Set model to evaluation mode

            
            # Load prompt templates for action selection
            with open("prontoqa_prompts/next_step_1shot.json", "r") as f:
                dic = json.load(f)
                inputs = dic["input"] + dic["facts_format"].format(" ".join(allowed_actions)) + dic["target_format"].format(query) + dic["claim_format"].format(last_state) + dic["next_step_prefix"] + " "
                
            # Encode input sequence
            input_ids = self.tokenizer.encode(inputs, return_tensors='pt').to(self.device)

            # Generate prefix embeddings for efficient processing
            prefix_output = self.model(input_ids[:, :-1], use_cache=True)
            prefix_past = prefix_output.past_key_values

            # Calculate log probabilities for each possible action
            action_logits = []
            for a in allowed_actions:
                # Encode and evaluate each action
                action_ids = self.tokenizer.encode(a, add_special_tokens=False,return_tensors='pt').to(self.device)
                input_ids_with_action = torch.cat([input_ids[:, -1:], action_ids], dim=-1)
                outputs = self.model(input_ids_with_action, past_key_values=prefix_past, use_cache=True)
                logits = outputs.logits  

                # Calculate token-by-token log probabilities
                total_log_prob = torch.zeros(1).to("cuda:0")
                for i in range(1, input_ids_with_action.shape[-1]):
                    probs = torch.softmax(logits[:, i - 1, :], dim=-1)
                    for j in range(1):
                        total_log_prob[j] += torch.log(probs[j, input_ids_with_action[j, i]])

                # Normalize by sequence length
                num_tokens = input_ids_with_action.shape[-1] - 1
                avg_log_prob = total_log_prob / num_tokens
                action_logits.append(avg_log_prob)

            # Apply temperature scaling and convert to probabilities
            action_logits = torch.stack(action_logits) / temperature
            action_logits = action_logits.to(torch.float32)
            probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))
            print("probabilities shape\n", probabilities.shape)

            # Select action based on probabilities
            idx = probabilities.argmax()
            print("last_state:\n", last_state)
            print("action space:\n", allowed_actions)
            print("action_idx:\n",idx)
            if not argmax:
                dist = Categorical(probs=probabilities.t())
                idx = dist.sample()

            action = allowed_actions[idx]

            # Remove selected action from available actions
            allowed_actions.remove(action)

            # Determine next state using cached transitions or generating new ones
            if last_state in self.transitions and action in self.transitions[last_state]:
                new_state = self.transitions[last_state][action]
            else:
                # Generate new state transition using language model
                with open("prontoqa_prompts/state_transit_examples_long.json", "r") as f:
                    dic = json.load(f)
                    world_update_prompt = dic["input"] + dic["facts_format"].format(last_state, action) + dic["next_claim_prefix"] + " "

                # Switch to base model for state transition
                lora_to_base(self.model)

                # Retry logic for state generation
                while True:
                    try:
                        world_output = query_LM(self.model, self.tokenizer, world_update_prompt, do_sample=False, num_return_sequences=1,
                                            eos_token_id=eos_token_id)[0][len(world_update_prompt):].strip()
                        new_state = world_output.split("\nClaim")[0].strip()
                        break
                    except Exception as e:
                        print(e)
                        print("An error occurred: Query LM fails, line 721")
                        time.sleep(1)

                print("old_state:\n", last_state)
                print("action:\n", action)
                print("new_state:\n", new_state)
                
                # Cache new transition
                if last_state not in self.transitions:
                    self.transitions[last_state] = {
                        action: new_state
                    }
                elif action not in self.transitions[last_state]:
                    self.transitions[last_state][action] = new_state

            # Check if goal state is reached
            finish = is_finish(new_state, query)

            
            # Test mode logic
            if (not finish) and ((step* 2 + 1) < len(plan)) and (not action in plan[step* 2 + 1][0]):
                finish = True
            states.append(last_state)
            actions.append(action)
            step += 1
            last_state = new_state

        # Add final state and calculate reward
        states.append(last_state)
        r1 = get_full_reward(plan, actions, self.sum_avg)

        return None, actions, states, r1, None


def lora_to_base(model):
    # Disable LoRA adapter layers in the model and set to evaluation mode.
    try:
        model.base_model.disable_adapter_layers()
    except:
        print("No adapter layers to disable")
    model.eval()
    
def base_to_lora(model):
    # Enable LoRA adapter layers in the model and set to training mode.
    try:
        model.base_model.enable_adapter_layers()
    except:
        print("No adapter layers to enable")
    model.train()


def blocksworld_planning(
    model,
    tokenizer,
    buffer_size=50,
    epochs=40,
):
    # Initialize replay buffer with direct buffer size
    rbuffer = ReplayBuffer(buffer_size=buffer_size)
    
    # Initialize logZ parameter
    logZ = torch.nn.Parameter(torch.tensor([0], dtype=torch.float))
    
    # Initialize data module with direct parameters
    data = PromptDataModule(
        prompt_file="prontoqa_prompts/next_step_examples.json",  # custom prompt file
        data_file="prontoqa_data/345hop_random_true.json",  # custom data file
        cot_file="prontoqa_data/data.txt",  # custom chain of thought file
    )
    
    # Get data splits
    train_probes = data.train_data
    val_probes = data.val_data
    test_probes = data.test_data
    
    # Initialize task with direct parameters
    task = BlocksWorldGFNTask(
        model=model,
        tokenizer=tokenizer,
        replay_buffer=rbuffer,
        train_data=train_probes,
        val_data=val_probes,
        test_data=test_probes
    )

    # Create a logger
    
    logger = TensorBoardLogger("logs", name="prontoqa_experiment")
    
    # Initialize trainer with direct parameters
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=epochs,
        accumulate_grad_batches=10,
        logger=logger
    )
    # Train the model
    trainer.validate(model=task, datamodule=data)





with open(f'temp={args.temp},{nametext}.txt', 'a') as f:
    with redirect_stdout(f), redirect_stderr(f):
        blocksworld_planning(
            model=model,
            tokenizer=tokenizer,
            buffer_size=50,  # Using default value as per code
            epochs=40,         # Using default value as per code
        )


    