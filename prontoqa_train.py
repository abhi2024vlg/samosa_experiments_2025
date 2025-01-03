import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import json
import os
from pytorch_lightning.loggers import TensorBoardLogger
from transformers.trainer_pt_utils import LabelSmoother
from torch.distributions import Categorical
import csv
import pytorch_lightning as pl
from contextlib import redirect_stdout, redirect_stderr

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

def load_model(
    pretrained_model,
    device,
    use_lora=False,
    use_4bit = False,
    test_only=False,
    load_checkpoint_path=None
):
    """
    Load and configure a model with optional 4-bit quantization and LoRA.
    
    Args:
        pretrained_model: Name or path of the pretrained model
        device: The device to load the model on
        use_lora: Whether to apply LoRA
        test_only: Whether this is for testing only
        load_checkpoint_path: Path to load checkpoint from (for testing)
    """

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                trust_remote_code=True,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

    else :
        model = AutoModelForCausalLM.from_pretrained(
        pretrained_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0"
        )
        
        model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model, 
        add_bos_token=False
    )

    if use_lora and not test_only:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer

    if test_only and load_checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, load_checkpoint_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
    

# Using LoRA for training
model, tokenizer = load_model(
    pretrained_model="meta-llama/Llama-3.2-1B",
    device=device,
    use_lora=True,
    use_4bit=True,
    test_only=False,
    load_checkpoint_path=None
)

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

class PromptDataModule(LightningDataModule):
    """
    This module handles the loading and preprocessing of prompts, questions, and chain-of-thought
    reasoning examples from the ProntoQA dataset. It supports filtering data based on specific
    keywords and maintains separate sets for training, validation, and testing.

    Args:
        prompt_file (str): Path to the JSON file containing base prompts
        data_file (str): Path to the JSON file containing main training data
        cot_file (str): Path to the file containing chain-of-thought examples
        train_size (int): Number of in-distribution examples for training. Defaults to 20
        val_size (int): Number of in-distribution examples for validation. Defaults to 10
        test_size (int): Number of out-of-distribution examples for testing. Defaults to 10
    """
    def __init__(
        self,
        prompt_file,
        data_file,
        cot_file,
        train_size=20,    # Changed default to 20
        val_size=10,
        test_size=10
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")

        # Load the base prompt template from JSON file
        with open(prompt_file, 'r') as f:
            self.base_prompt = json.load(f)["input"]
        
        # Initialize core components
        self.train_size = train_size  # Changed from max_size to train_size
        self.val_size = val_size
        self.test_size = test_size
        
        # Initialize data storage
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        # Store file paths for data loading
        self.data_file = data_file
        self.cot_file = cot_file
        
        # Define keywords for data filtering
        # In-distribution keywords typically related to mathematical concepts
        self.indis_keywords = ["3", "31", "131071", "real", "number", "imaginary", "numbers"]
        # Out-of-distribution keywords typically related to biological concepts
        self.ood_keywords = ["bony", "insect", "cold-blooded", "animal"]

    def setup(self, stage):
        """
        Performs the data setup process including loading, filtering, and splitting the dataset.
        
        This method ensures:
        1. 20 in-distribution examples for training
        2. 10 in-distribution examples for validation
        3. 10 out-of-distribution examples for testing
        
        Args:
            stage: Current stage of training ('fit', 'validate', 'test')
        """
        
        in_dist_data = []  # Store all in-distribution examples
        ood_data = []      # Store all out-of-distribution examples
        potential_train_data = []  # Create a list for potential training examples
        
        # Load and process main dataset
        with open(self.data_file, 'r') as f:
            all_data = json.load(f)

        # Process each example in the dataset
        for key in all_data:
            if key.startswith("example"):
                example = all_data[key]["test_example"]
                # Extract components of each example
                actions = example["question"]
                query = example["query"]
                plan = example["chain_of_thought"]
                gt = example["answer"]
                example_data = [actions, query, plan, gt]
                
                # Filter examples based on keywords
                if any(keyword in actions for keyword in self.ood_keywords):
                    ood_data.append(example_data)
                elif any(keyword in actions for keyword in self.indis_keywords):
                    in_dist_data.append(example_data)
                    potential_train_data.append(example_data)

        # Process chain-of-thought examples
        train_answer = []
        valid_answer = []
        
        # Parse chain-of-thought file and separate correct/incorrect examples
        with open(self.cot_file, "r") as f:
            for line in f.read().splitlines():
                # Extract answer from the line
                answer_start = line.index("answer='") + len("answer='")
                answer_end = line.index("';")
                answer = line[answer_start:answer_end]
                
                # Sort answers based on correctness
                if "correct=False" in line:
                    train_answer.append(answer)
                else:
                    valid_answer.append(answer)

        # Match chain-of-thought answers with training examples
        # Continue matching until we have exactly train_size examples
        for ans in train_answer:
            if len(self.train_data) >= self.train_size:
                break
                
            actions = ans.split("\\n")
            i = len(actions) - 1
            for train_ex in potential_train_data:
                if train_ex in self.train_data:
                    continue
                    
                start_id = 0
                # Match actions backwards from the end
                while i >= 0 and train_ex[2][-1-start_id*2] == actions[i]:
                    start_id += 1
                    i -= 1
                if i == -1:
                    self.train_data.append(train_ex)
                    break

        # Ensure we have exactly train_size training examples
        remaining_in_dist = [x for x in in_dist_data if x not in self.train_data]
        if len(self.train_data) < self.train_size:
            additional_needed = self.train_size - len(self.train_data)
            self.train_data.extend(remaining_in_dist[:additional_needed])
            remaining_in_dist = remaining_in_dist[additional_needed:]

        # Create validation set (exactly val_size in-distribution examples)
        self.val_data = remaining_in_dist[:self.val_size]
        
        # Create test set (exactly test_size out-of-distribution examples)
        self.test_data = ood_data[:self.test_size]

        # Print dataset statistics
        print(f"Number of training data: {len(self.train_data)}")
        print(f"Number of validation data: {len(self.val_data)}")
        print(f"Number of test data: {len(self.test_data)}")

    def train_dataloader(self):
        """Returns DataLoader for training data with shuffling enabled."""
        return DataLoader(self.train_data, shuffle=True, batch_size=1)

    def val_dataloader(self):
        """Returns DataLoader for validation data."""
        return DataLoader(self.val_data, batch_size=1)

    def test_dataloader(self):
        """Returns DataLoader for test data."""
        return DataLoader(self.test_data, batch_size=1)

def lora_to_base(model):
    """
    Disable LoRA adapter layers in the model and set to evaluation mode.
    
    Args:
        model: The neural network model with LoRA layers
    """
    try:
        model.base_model.disable_adapter_layers()
    except:
        print("No adapter layers to disable")
    model.eval()
    
def base_to_lora(model):
    """
    Enable LoRA adapter layers in the model and set to training mode.
    
    Args:
        model: The neural network model with LoRA layers
    """
    try:
        model.base_model.enable_adapter_layers()
    except:
        print("No adapter layers to enable")
    model.train()

def tb_loss(log_pf, log_r, logz, log_bf=None, logpartition=True):
    
    """
    Calculate the trajectory balance loss with optional partition function.
    
    This version supports both traditional and partition-normalized loss calculations.
    
    Args:
        log_pf (torch.Tensor): Log forward probabilities
        log_r (torch.Tensor): Log rewards
        logz (torch.Tensor): Log partition function
        log_bf (torch.Tensor, optional): Log backward probabilities
        logpartition (bool): Whether to use partition-normalized loss calculation
    
    Returns:
        torch.Tensor: Computed mean loss value
    """
    
    print("log_pf: ", log_pf)
    print("log_r: ", log_r)
    print("logz: ", logz)
    if logpartition:
        if log_bf != None:
            scores = log_pf - log_r - log_bf
            loss = (scores - scores.mean()) ** 2 
        else:
            scores = log_pf - log_r
            loss = (scores - scores.mean()) ** 2 
    else:
        if log_bf != None:
            loss = (log_pf + logz - log_r - log_bf) ** 2
        else:
            loss = (log_pf + logz - log_r) ** 2
    return loss.mean()

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
        logZ,
        tokenizer,
        replay_buffer,
        train_data=None,
        val_data=None,
        test_data=None,
        use_lora=True,
        epochs=40,
        n_samples=4,
        lr=5e-6,
        logZ_lr=1e-1
    ):
        """
        Initialize the BlocksWorld GFlowNet task.

        Args:
            model: Language model for generating actions and transitions
            logZ: Partition function estimate
            tokenizer: Tokenizer for the language model
            replay_buffer: Buffer for storing and sampling trajectories
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            use_lora: Whether to use LoRA adaptation
            n_samples: Number of trajectories to sample per step
            lr: Learning rate for model parameters
            logZ_lr: Learning rate for logZ parameter
        """
        super().__init__()

        self.sum_avg="sum"
        self.epochs=epochs
        self.batch_size = 1
        self.use_buffer_prob = 0.5
        ## Storing 
        self.test_csv = "test_success.csv"
        self.valid_csv = "validation_success.csv"
        # Store model and parameters
        self.logZ = logZ
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

        # Store hyperparameters
        self.n_samples = n_samples 
        self.lr = lr
        self.logZ_lr = logZ_lr
        self.reward_temp_start = 1
        self.reward_temp_end = 2
        self.pf_temp_end = 1.0
        self.pf_temp_prob = 0.5
        self.pf_temp_start = 2.0
        self.epsilon_start = 0.4
        self.epsilon_end = 0.01
        self.ll_weight=0.9
        self.step = 2
        self.use_4bit = True
        
        # Define learning rate schedule
        self.get_lr_at_step = lambda step: min(step / 20 * self.lr, self.lr)

        
        
        # Define reward temperature schedule
        self.get_reward_temp_at_step = lambda step: self.reward_temp_start + (
           self.reward_temp_end - self.reward_temp_start
        ) * 1
        
        
        # Initialize caches and tracking
        self.ignore_token_id = LabelSmoother.ignore_index

        self.reward_temperature = self.reward_temp_start

        
        self.pf_temperature = self.pf_temp_start

        
        self.epsilon = self.epsilon_start

        # Load or initialize transition cache
        transition_path = f"transitions/{self.step}/transition.json"
        self.wrong_transitions = {}
        self.ls_wrong_transitions = {}

        if os.path.exists(transition_path):
            with open(transition_path, 'r') as f:
                self.transitions = json.load(f)
        else:
            self.transitions = {}

    def forward(self, problem, pf_temperature=1.0):
        """
        Forward pass to generate a trajectory for a given problem.
        
        Args:
            problem: Tuple of (actions, query, plan, ground_truth)
            pf_temperature: Temperature parameter for forward policy
            
        Returns:
            generated_text: Generated trajectory text
            actions: List of actions taken
            states: List of states encountered
            reward: Trajectory reward
            sample: Sample information
        """
        # Unpack problem components
        ACTIONS, QUERY, PLAN, GT = problem
        GT = GT[0]
        ACTIONS = ACTIONS[0]
        QUERY = QUERY[0]

        # Generate trajectory using temperature-based sampling

        (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
        ) = self.generate_trajectories_v2(
            query = QUERY,
            allowed_actions = ACTIONS,
            gt = GT,
            plan = PLAN,
            temperature=pf_temperature,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0]
                            )

        return generated_text, actions, states, sample, reward

    def training_step(self, problem, batch_idx):
        """
        Execute a single training step.
        
        Implements the GFlowNet training loop:
        1. Sample trajectories from current policy
        2. Compute rewards and log probabilities
        3. Update model using trajectory balance loss
        
        Args:
            problem: Current training problem
            batch_idx: Index of current batch
            
        Returns:
            loss: Training loss for this step
        """
        # Reset wrong transitions tracking
        self.wrong_transitions = {}

        # Unpack problem components
        ACTIONS, QUERY, PLAN, GT = problem
        ACTIONS = ACTIONS[0]
        QUERY = QUERY[0]
        GT = GT[0]
      
        ########################## Compute the reward for ground-truth trajectory ##########################

        # Initialize lists to store trajectory information
        LOG_R = []  # Log rewards
        LOG_PF = []  # Forward policy log probabilities
        LOG_BF = []  # Backward policy log probabilities

        # Sample trajectories either from replay buffer or by generating new ones
        if (
            random.random() < self.use_buffer_prob
            and self.replay_buffer.sample(self.n_samples, QUERY)[0] is not None
        ):
            # Use samples from replay buffer
            (log_reward_list,
            state_list,
            sample_list
            ) = self.replay_buffer.sample(
                self.n_samples, QUERY
            )

            # Compute forward and backward probabilities for sampled trajectories
            for state, sample in zip(state_list, sample_list):
                (actions, states) = eval(state)
                log_pf, log_bf = self.forward_prob(QUERY, ACTIONS, actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)
            LOG_R.extend(log_reward_list)
        
        else:
            # Generate new trajectories
            best_actions = None
            best_states = None
            best_reward = -9999

            # Generate multiple trajectories and keep track of the best one
            for _ in range(self.n_samples):
                # Sample temperature
                if np.random.rand() < self.pf_temp_prob:
                    pf_temp = self.pf_temperature
                else:
                    pf_temp = 0.7

                # Generate trajectory
                generated_text, actions, states, sample, reward = self.forward(
                    problem, pf_temp
                )

                # Compute language model based reward
                ll_reward = self.get_ll_reward_rule_hard(actions, states, QUERY)

                # Calculate final reward based on reward aggregation method
                if self.sum_avg == "sum":
                    log_r = torch.log(self.ll_weight * ll_reward.sum())
                    LOG_R.append(log_r)
        
                elif self.sum_avg == "avg":
                    log_r = torch.log(self.ll_weight * ll_reward.mean())
                    LOG_R.append(log_r)

                
                print("generated ll: \n",  ll_reward)
                print("trajectory query: \n",  QUERY)
                print("trajectory states: \n",  states)
                print("trajectory actions: \n",  actions)
                
                # Store trajectory in replay buffer
                generated_text = (actions, states)
                self.replay_buffer.add(QUERY, str(generated_text), sample, log_r)

                # Compute forward and backward probabilities
                log_pf, log_bf = self.forward_prob(QUERY, ACTIONS, actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)

                # Update best trajectory if current one is better
                if log_r > best_reward:
                    best_actions  = actions
                    best_states = states
                    best_reward = log_r

            # Perform local search to improve best trajectory
            self.ls_wrong_transitions = {}
            for _ in range(6):
                _, actions, states, reward, _ = self.local_search(
                        query = QUERY,
                        allowed_actions = ACTIONS,
                        gt_plan= PLAN,
                        past_actions = best_actions,
                        eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0])

                # Compute reward for locally searched trajectory
                ll_reward = self.get_ll_reward_rule_hard(actions, states, QUERY)
                print("generated ll_ls: \n",  ll_reward)
                print("trajectory query_ls: \n",  QUERY)
                print("trajectory states_ls: \n",  states)
                print("trajectory actions_ls: \n",  actions)

                
                if self.sum_avg == "sum":
                    log_r = torch.log(self.ll_weight * ll_reward.sum())
        
                elif self.sum_avg == "avg":
                    log_r = torch.log(self.ll_weight * ll_reward.mean())
                    
                # If local search found better trajectory, add it to samples
                if log_r > best_reward:
                    LOG_R.append(log_r)
                    generated_text = (actions, states)
                    self.replay_buffer.add(QUERY, str(generated_text), sample, log_r)
                    log_pf, log_bf = self.forward_prob(QUERY, ACTIONS, actions, states)
                    LOG_PF.append(log_pf)
                    LOG_BF.append(log_bf)

            self.ls_wrong_transitions = {}

        # Convert lists to tensors and move to device
        LOG_PF = torch.stack(LOG_PF).to(self.model.device)
        LOG_BF = torch.stack(LOG_BF).to(self.model.device)
        LOG_R = torch.stack(LOG_R).to(self.model.device)

        # Apply reward temperature
        LOG_R = LOG_R * (1 / self.reward_temperature)

        # Ensure model is in LoRA mode for training
        base_to_lora(self.model)

        # Reset wrong transitions tracking
        self.wrong_transitions = {}
        
        # Get the Trajectory balance loss
        loss = tb_loss(
            log_pf=LOG_PF,
            log_r=LOG_R,
            logz=self.logZ,
            log_bf=None
        )

        # Log metrics
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        self.log(
            "train/logR",
            LOG_R.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size
        )

        return loss

    @torch.no_grad()
    def test_step(self, problem, batch_idx):
        """
        Execute evaluation step during testing phase.
        
        This method evaluates the model's performance by:
        1. Generating multiple trajectories for each test problem
        2. Checking if solutions reach the target state
        3. Verifying if the solution follows the planned proof steps
        4. Recording successful solutions and their variations
        
        Args:
            problem: Current test problem containing actions, query, plan, and ground truth
            batch_idx: Index of current batch
            
        Returns:
            None, but logs various metrics including success rate and solution count
        """
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

        # First try with argmax sampling (deterministic)
        (
        generated_text, 
        actions, 
        states,
        reward, 
        sample
        ) = self.generate_trajectories_v2(
            query = QUERY,
            allowed_actions = ACTIONS,
            gt = GT,
            plan = PLAN,
            temperature=0.5,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
            mode="test",
            argmax=True
        )
        # Check if solution reaches target state
        if eval_tf(states[-1], QUERY, GT):
            total_success += 1
            actions_joined = '\n'.join(actions)
            if actions_joined not in success_text:
                success_text.append((QUERY, actions_joined))

        # Verify proof steps
        last_3_plans = [PLAN[-5][0], PLAN[-3][0],PLAN[-1][0]]
        if "Finish" not in actions[-1]:
            last_3_actions = actions[-3:]
        else:
            last_3_actions = actions[-4:-1]

        if last_3_actions == last_3_plans:
            total_proof_success += 1

        # Try multiple random samples to find different solutions
        for _ in range(32):

            (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
            ) = self.generate_trajectories_v2(
                query = QUERY,
                allowed_actions = ACTIONS,
                gt = GT,
                plan = PLAN,
                temperature=0.5,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )

            # Track successful solutions
            if eval_tf(states[-1], QUERY, GT):
                total_success += 1
                actions_joined = '\n'.join(actions)
                if actions_joined not in success_text:
                    success_text.append((QUERY, actions_joined))

            # Verify proof steps
            last_3_plans = [PLAN[-5][0], PLAN[-3][0],PLAN[-1][0]]

            if "Finish" not in actions[-1]:
                last_3_actions = actions[-3:]
            else:
                last_3_actions = actions[-4:-1]

            if last_3_actions == last_3_plans:
                total_proof_success += 1

        # Save successful solutions to CSV
        with open(self.test_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(success_text)

        # Convert counts to binary success indicators
        if total_success > 0:
            success = 1
        else:
            success = 0
        if total_proof_success > 0:
            psuccess = 1
        else:
            psuccess = 0

        # Log metrics
        self.log(
            "test/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        self.log(
            "test/n_solutsion",
            len(success_text),
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
        self.log(
            "test/n_psolutsion",
            total_proof_success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size
        )
        

    @torch.no_grad()
    def validation_step(self, problem, batch_idx):
        # pass
        self.wrong_transitions = {}
        base_to_lora(self.model)    
        self.model.eval()       

        ACTIONS, QUERY, PLAN, GT = problem

        ACTIONS = ACTIONS[0]
        QUERY = QUERY[0]
        GT = GT[0]

        total_success = 0
        total_proof_success = 0
        success_text = []

        #argmax

        (
        generated_text, 
        actions, 
        states,
        reward, 
        sample
        ) = self.generate_trajectories_v2(
            query = QUERY,
            allowed_actions = ACTIONS,
            gt = GT,
            plan = PLAN,
            temperature=0.5,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
            mode="test",
            argmax=True
        )

        if eval_tf(states[-1], QUERY, GT):
            total_success += 1
            s = "success"
        else:
            s = "fail"
        actions_joined = '==>'.join(actions)
        states_joined = '==>>'.join(states)

        #if actions_joined not in success_text:
        last_3_plans = [PLAN[-6][0], PLAN[-4][0],PLAN[-2][0]]
        if "Finish" not in actions[-1]:
            last_3_actions = actions[-3:]
        else:
            last_3_actions = actions[-4:-1]

        if last_3_actions == last_3_plans:
            total_proof_success += 1
            ps = "proof_success"
        else:
            ps = "proof_fail"

        success_text.append((s, ps, QUERY, GT, actions_joined, states_joined))

        for _ in range(32):

            (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
            ) = self.generate_trajectories_v2(
                query = QUERY,
                allowed_actions = ACTIONS,
                gt = GT,
                plan = PLAN,
                temperature=0.5,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )

            if eval_tf(states[-1], QUERY, GT):
                total_success += 1
                s = "success"
            else:
                s = "fail"
            actions_joined = '==>'.join(actions)
            states_joined = '==>>'.join(states)

            #if actions_joined not in success_text:
            last_3_plans = [PLAN[-6][0], PLAN[-4][0],PLAN[-2][0]]
            if "Finish" not in actions[-1]:
                last_3_actions = actions[-3:]
            else:
                last_3_actions = actions[-4:-1]

            if last_3_actions == last_3_plans:
                total_proof_success += 1
                ps = "proof_success"
            else:
                ps = "proof_fail"

            success_text.append((s, ps, QUERY, GT, actions_joined, states_joined))

        with open(self.valid_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(success_text)

        if total_success > 0:
            success = 1
        else:
            success = 0


        if total_proof_success > 0:
            psuccess = 1
        else:
            psuccess = 0
        
        self.wrong_transitions = {}

        self.log("val/success",success,sync_dist=True,prog_bar=True,batch_size=self.batch_size)
        
        self.log("val/n_solutsion",total_success,sync_dist=True,prog_bar=True,batch_size=self.batch_size)

        self.log("val/psuccess",psuccess,sync_dist=True,prog_bar=True,batch_size=self.batch_size)
        
        self.log("val/n_psolutsion",total_proof_success,sync_dist=True,prog_bar=True,batch_size=self.batch_size)

    def on_train_epoch_start(self):
        # Log scheduled quantities
        current_epoch = self.trainer.current_epoch
        if (current_epoch + 1) % 6 == 0:
            self.pf_temperature = self.pf_temp_start - (self.pf_temp_start - self.pf_temp_end) / (self.epochs // 6)

        if current_epoch < self.epochs // 2:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) / (self.epochs // 2)
        
        if current_epoch < self.epochs // 2:
            self.reward_temperature = self.reward_temp_start + current_epoch * (self.reward_temp_end - self.reward_temp_start) / (self.epochs // 2)
    
        self.log("scheduled/R_temperature", self.reward_temperature, sync_dist=True)

    def configure_optimizers(self):
        if self.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            optimizer = bnb.optim.PagedAdamW8bit([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5),
                "monitor": "metric_to_track",
                "frequency": 10,
            }
            }
        else:
            return torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])
            
    def local_search(self,
                    query,
                    allowed_actions,
                    gt_plan,
                    past_actions,
                    eos_token_id,
                    max_steps=10,
                    mode="train",
                          ):
        """
        Perform local search to improve a trajectory.
        
        This method attempts to improve an existing trajectory by:
        1. Making small modifications to the action sequence
        2. Evaluating the modified trajectories
        3. Accepting improvements when found
        
        Args:
            query: Target state description
            allowed_actions: List of possible actions
            gt_plan: Ground truth plan
            past_actions: Previous action sequence to improve
            eos_token_id: End of sequence token ID
            max_steps: Maximum number of modification steps
            mode: "train" or "test" mode
            
        Returns:
            Tuple of (None, actions, states, reward, None) for the best trajectory found
        """
        print("local search starts!!!")

        # Parse allowed actions
        allowed_actions = allowed_actions.split(". ")
        print("allowed_actions:")
        print(allowed_actions)
        initial_state = allowed_actions[-1]
        allowed_actions = [a+"." for a in allowed_actions[:-1]]

        last_state = initial_state
        print("last_state:\n", last_state)

        # Initialize trajectory tracking
        actions = []
        finish = False
        step = 0
        states = []

        # Generate modified trajectory
        while not finish and (step <= max(len(gt_plan)+1, max_steps)) and len(allowed_actions) > 0:
            # Choose action: either keep past action or try new one
            if step < len(past_actions)-1:
                action = past_actions[step]
            else:
                allowed_actions_ = [act for act in allowed_actions if act not in actions]
                if len(allowed_actions_) != 0:
                    action = random.choice(allowed_actions_)
                else:
                    action = random.choice(allowed_actions)

            allowed_actions.remove(action)

            # Get or compute next state
            if last_state in self.transitions and action in self.transitions[last_state]:
                new_state = self.transitions[last_state][action]
            else:
                # Generate next state using language model
                with open("state_transit_examples_long.json", "r") as f:
                    dic = json.load(f)
                    world_update_prompt = dic["input"] + dic["facts_format"].format(last_state, action) + dic["next_claim_prefix"] + " "

                lora_to_base(self.model)
                while True:
                    try:
                        world_output = query_LM(self.model, self.tokenizer, world_update_prompt, do_sample=False, num_return_sequences=1,
                                            eos_token_id=eos_token_id)[0][len(world_update_prompt):].strip()
                        new_state = world_output.split("\nClaim")[0].strip()
                        break
                    except Exception as e:
                        print(e)
                        print("An error occurred: Query LM fails, line 721")
                        import time
                        time.sleep(1)

                print("new_state222:\n", last_state)
                print("action222:\n", action)
                print("new_state222:\n", new_state)
                # Cache the transition
                if last_state not in self.transitions:
                    self.transitions[last_state] = {
                        action: new_state
                    }
                elif action not in self.transitions[last_state]:
                    self.transitions[last_state][action] = new_state

            # Check if goal is reached
            finish = is_finish(new_state, query)

            # Handle training vs testing mode differently
            if mode=="train":
                if (not finish) and ((step* 2 + 1) < len(gt_plan)) and (not action in gt_plan[step* 2 + 1][0]):
                    # Track wrong transitions during training
                    if last_state not in self.ls_wrong_transitions:
                        print("pass here!", action, step)
                        self.ls_wrong_transitions[last_state] = [action]
                        states.append(last_state)
                        actions.append(action)
                        last_state = new_state
                        finish = True
                    elif action not in self.ls_wrong_transitions[last_state] and (action not in [e[0] for e in gt_plan]):
                        print("pass here222!", action, step)
                        self.ls_wrong_transitions[last_state].append(action)
                        states.append(last_state)
                        actions.append(action)
                        last_state = new_state
                        finish = True
                    else:
                        print("known wrong pass here222!", action, step)

                else:
                    states.append(last_state)
                    actions.append(action)
                    step += 1
                    last_state = new_state
            else:
                # Simpler handling for test mode
                if (not finish) and ((step* 2 + 1) < len(gt_plan)) and (not action in gt_plan[step* 2 + 1][0]):
                    finish = True
                states.append(last_state)
                actions.append(action)
                step += 1
                last_state = new_state

        # Add final state
        states.append(last_state)

        # Compute reward for modified trajectory
        r1 = get_full_reward(gt_plan, actions, self.sum_avg)

        return None, actions, states, r1, None


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
        
        """
        Generates action trajectories for a given query using a language model with exploration capabilities.
        
        This method implements a trajectory generation algorithm that:
        1. Iteratively selects actions based on model predictions or random exploration
        2. Updates world state after each action using cached or newly generated transitions
        3. Evaluates trajectories against ground truth and planning constraints
        
        Args:
        query (str): The target query or goal state to achieve
        allowed_actions (str): Period-separated string of available actions
        gt (str): Ground truth or expected outcome
        plan (list): List of planned action sequences to follow
        temperature (float): Temperature parameter for controlling randomness in action selection
        eos_token_id (int): End of sequence token ID for the tokenizer
        max_steps (int, optional): Maximum number of steps to generate. Defaults to 10
        argmax (bool, optional): Whether to use greedy action selection. Defaults to False
        mode (str, optional): Operating mode ("train" or "test"). Defaults to "train"
        
        Returns:
        tuple: (None, actions list, states list, trajectory reward, None)
        - actions: List of selected actions in order
        - states: List of world states after each action
        - r1: Numerical reward based on plan adherence
        """
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

            # Epsilon-greedy exploration in training mode
            if np.random.rand() < self.epsilon and mode == "train":
                action = random.choice(allowed_actions)
            else:
                # Load prompt templates for action selection
                with open("next_step_1shot.json", "r") as f:
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
                with open("state_transit_examples_long.json", "r") as f:
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
                        import time
                        time.sleep(1)

                print("new_state222:\n", last_state)
                print("action222:\n", action)
                print("new_state222:\n", new_state)
                
                # Cache new transition
                if last_state not in self.transitions:
                    self.transitions[last_state] = {
                        action: new_state
                    }
                elif action not in self.transitions[last_state]:
                    self.transitions[last_state][action] = new_state

            # Check if goal state is reached
            finish = is_finish(new_state, query)

            # Handle training mode specific logic
            if mode=="train":
                # Check if action deviates from plan
                if (not finish) and ((step* 2 + 1) < len(plan)) and (not action in plan[step* 2 + 1][0]):
                    if last_state not in self.wrong_transitions:
                        print("pass here!", action, step)
                        self.wrong_transitions[last_state] = [action]
                        states.append(last_state)
                        actions.append(action)
                        last_state = new_state
                        finish = True
                    elif action not in self.wrong_transitions[last_state] and (action not in [e[0] for e in plan]):
                        print("pass here222!", action, step)
                        self.wrong_transitions[last_state].append(action)
                        states.append(last_state)
                        actions.append(action)
                        last_state = new_state
                        finish = True
                    else:
                        print("known wrong pass here222!", action, step)

                else:
                    states.append(last_state)
                    actions.append(action)
                    step += 1
                    last_state = new_state
            else:
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

    def get_ll_reward_rule_hard(self, actions, states, goal):
        """
        Compute language model based reward with strict matching criteria.
        
        This reward function encourages:
        1. Action selection that matches the current state
        2. Final state matching with the goal
        
        Args:
            actions: List of actions taken
            states: List of states encountered
            goal: Target state description
            
        Returns:
            torch.Tensor of rewards for each step
        """
        reward = [0] * len(states)

        for step_idx, (state, action) in enumerate(zip(states, actions)):
            intuition = 0.00001 # Small baseline reward

            # Only compute reward if previous step was successful or it's the first step
            if step_idx == 0 or reward[step_idx - 1] != 0.00001:
                if step_idx < len(actions) - 1:
                    next_state = states[step_idx+1]
                    # Reward for action matching current state
                    if state.replace(".", "").split(" ")[-1].replace("s", "").lower() in action.replace("s", "").lower():
                        if self.sum_avg=="sum":
                            intuition += 20
                        else:
                            intuition += 100
                else: 
                    # Reward for final state matching goal
                    if state.replace(".", "").split(" ")[-1].replace("s", "").lower() in goal.replace("s", "").lower():
                        if self.sum_avg=="sum":
                            intuition += 20
                        else:
                            intuition += 100

            reward[step_idx] = intuition

        return torch.tensor(reward).to(self.device)

    def find_best_match(self, string_list, target_string):
        """
        Finds the closest matching string from a list of strings by comparing word overlap with a target string.
        
        This method implements a simple similarity matching algorithm that works by:
        1. Converting strings to lowercase and removing periods
        2. Tokenizing strings into sets of words
        3. Counting common words between the target and each candidate string
        4. Selecting the candidate with the most word overlap
        
        Args:
        string_list (list): A list of candidate strings to search through
        target_string (str): The reference string to match against
        
        Returns:
        int: The index of the best matching string from string_list
        """
        # Convert target string to lowercase, remove periods, and split into a set of unique words
        # This normalization helps ensure consistent matching regardless of case or punctuation
        target_words = set(target_string.replace(".", "").lower().split())
        
        def count_common_words(entry):
            """
            Helper function that counts how many words two strings have in common.
             
            Args:
            entry (str): A candidate string to compare against the target
             
            Returns:
            int: Number of words in common between entry and target_string
            """
            entry_words = set(entry.replace(".", "").lower().split())
            return len(target_words.intersection(entry_words))
        
         # Calculate overlap scores for all candidates and find the best match
        common_counts = [count_common_words(entry) for entry in string_list]
        best_index = common_counts.index(max(common_counts))
        
        return best_index

    def forward_prob(self, query, allowed_actions, actions, states):
        """
        Calculates forward and backward probabilities for a sequence of actions in a language model.
        
        This method computes two probability distributions:
        1. Forward probability: The likelihood of each action given the current state and query
        2. Backward probability: A uniform distribution over allowed actions
        
        The method processes actions sequentially, maintaining state information and updating
        probabilities based on the language model's predictions.
        
        Args:
        query (str): The input query or instruction
        allowed_actions (str): Period-separated string of permitted actions
        actions (list): Sequence of actions to evaluate
        states (list): Sequence of states corresponding to the actions
        
        Returns:
        tuple: (sum of log forward probabilities, sum of log backward probabilities)
        
        Notes:
        - Uses a template-based approach for formatting inputs
        - Handles LoRA model adaptation if enabled
        - Implements batched processing for efficient computation
        - Includes fallback matching for inexact action matches
        """
        # Enable LoRA adaptations if specified in arguments
        if  use_lora:
            base_to_lora(self.model)

        # Preprocess allowed actions by splitting on periods and reformatting
        allowed_actions = allowed_actions.split(". ")
        allowed_actions = allowed_actions[:-1]  # Remove empty string after last period
        allowed_actions = [a+"." for a in allowed_actions]

        print("forward_prob_actions!!!:\n", actions)

        # Initialize state tracking
    
        initial_state = states[0]
        last_state = initial_state
        log_pf = []  # Forward log probabilities
        log_bf = []  # Backward log probabilities

        # Load prompt templates from configuration file
        with open("next_step_1shot.json", "r") as f:
            dic = json.load(f)
            # Construct base input template with allowed actions
            inputs_template = dic["input"] + dic["facts_format"].format(" ".join(allowed_actions))

        # Process each action in sequence
        for step in range(len(actions)):
            # Construct full input by combining template, query, current state, and prefix
            inputs = inputs_template + dic["target_format"].format(query) + dic["claim_format"].format(last_state) + dic["next_step_prefix"] + " "

            # Convert input text to token IDs
            input_ids = self.tokenizer.encode(inputs, return_tensors='pt').to("cuda:0")
            action = actions[step]

            # Prepare batch of possible actions for parallel processing
            bsz = len(allowed_actions)  
            action_texts = [ac for ac in allowed_actions]
            action_ids = [self.tokenizer.encode(a, add_special_tokens=False, return_tensors='pt').to("cuda:0") for a in action_texts]

            # Pad action sequences to same length for batch processing
            max_length = max(len(aid[0]) for aid in action_ids)
            padded_action_ids = [torch.cat([aid, torch.full((1, max_length - len(aid[0])), self.tokenizer.pad_token_id, device=self.device)], dim=-1) for aid in action_ids]
            
            # Combine input with each possible action for batch processing
            batch_input_ids_with_actions = torch.cat([torch.cat([input_ids, pid], dim=-1) for pid in padded_action_ids], dim=0)

            # Get model predictions
            batch_outputs = self.model(batch_input_ids_with_actions, use_cache=True)
            batch_logits = batch_outputs.logits
            
            # Calculate token-by-token probabilities for each action
            total_log_prob = torch.zeros(bsz).cuda()
            for i in range(input_ids.shape[-1], batch_input_ids_with_actions.shape[-1]):
                probs = torch.softmax(batch_logits[:, i - 1, :], dim=-1)
                for j in range(bsz):
                    if batch_input_ids_with_actions[j, i] != self.tokenizer.pad_token_id:
                        total_log_prob[j] += torch.log(probs[j, batch_input_ids_with_actions[j, i]])
            action_logits = total_log_prob

            # Convert logits to probabilities
            action_logits = action_logits.to(torch.float32)
            probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))

            # Find index of current action, falling back to fuzzy matching if exact match fails
            try:
                idx = allowed_actions.index(action)
            except:
                print("execute find best match:\n", allowed_actions, action)
                idx = self.find_best_match(allowed_actions, action)

            # Record probabilities and update state
            log_pf.append(torch.log(probabilities[idx]))
            last_state = states[step+1]

            # Calculate uniform backward probability
            pb = torch.tensor(1 / len(allowed_actions))
            log_bf.append(torch.log(pb))

        # Return sums of log probabilities
        return torch.stack(log_pf).sum(), torch.stack(log_bf).sum()

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
        prompt_file="next_step_examples.json",  # custom prompt file
        data_file="345hop_random_true.json",  # custom data file
        cot_file="data.txt",  # custom chain of thought file
    )
    
    # Get data splits
    train_probes = data.train_data
    val_probes = data.val_data
    test_probes = data.test_data
    
    # Initialize task with direct parameters
    task = BlocksWorldGFNTask(
        model=model,
        logZ=logZ,
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
        precision=16,
        max_epochs=epochs,
        accumulate_grad_batches=10,
        logger=logger
    )
    # Train the model
    trainer.fit(model=task, datamodule=data)

from contextlib import redirect_stdout, redirect_stderr

with open('training_log.txt', 'a') as f:
    with redirect_stdout(f), redirect_stderr(f):
        blocksworld_planning(
            model=model,
            tokenizer=tokenizer,
            buffer_size=50,  # Using default value as per code
            epochs=40,         # Using default value as per code
        )

with open('training_log.txt', 'a') as f:
    with redirect_stdout(f), redirect_stderr(f):
        blocksworld_planning(
            model=model,
            tokenizer=tokenizer,
            buffer_size=50,  # Using default value as per code
            epochs=40,         # Using default value as per code
        )
