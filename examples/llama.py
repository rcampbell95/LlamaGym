import os
import numpy as np

from llamagym import Agent

from huggingface_hub import login

from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from peft import LoraConfig
from peft import LoraConfig

import wandb
from tqdm import trange

import gymnasium as gym


class BlackjackAgent(Agent):
    def get_system_prompt(self) -> str:
        return """You are an expert blackjack player. Every turn, you'll see your current sum, the dealer's showing card value, and whether you have a usable ace. Win by exceeding the dealer's hand but not exceeding 21.
Decide whether to stay with your current sum by writing "Action: 0" or accept another card by writing "Action: 1". Accept a card unless very close to 21."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"You: {observation[0]}. Dealer: {observation[1]}. You have {'an' if bool(observation[2]) else 'no'} ace."

    def extract_action(self, response: str) -> gym.core.ActType:
        match = re.compile(r"Action: (\d)").search(response)
        if match:
            return int(match.group(1))

        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("0", "1"):
            if "stick" in response.lower():
                return 0
            elif "hit" in response.lower():
                return 1

        return 0

device = "cpu"

HF_TOKEN = os.environ.get("HF_TOKEN")

hyperparams = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "env": "Blackjack-v1",
    "lora/r": 16,
    "lora/lora_alpha": 32,
    "lora/lora_dropout": 0.05,
    "lora/bias": "none",
    "lora/task_type": "CAUSAL_LM",
    "load_in_8bit": False,
    "batch_size": 8,
    "seed": 42069,
    "episodes": 1000,
    "generate/max_new_tokens": 32,
    "generate/do_sample": True,
    "generate/top_p": 0.6,
    "generate/top_k": 0,
    "generate/temperature": 0.9,
}

lora_config = LoraConfig(
    **{
        key.split("/")[-1]: value
        for key, value in hyperparams.items()
        if key.startswith("lora/")
    }
)

wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    hyperparams["model_name"],
    peft_config=lora_config,
    load_in_8bit=hyperparams["load_in_8bit"],
    token=HF_TOKEN).to(device)
tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"])
agent = BlackjackAgent(model, tokenizer, device)

agent = BlackjackAgent(
    model,
    tokenizer,
    device,
    {
        key: value
        for key, value in hyperparams.items()
        if key.startswith("generate/")
    },
    {
        "batch_size": hyperparams["batch_size"],
        "mini_batch_size": hyperparams["batch_size"],
    },
)
env = gym.make(hyperparams["env"], natural=False, sab=False)
epsilon = .2

for episode in trange(hyperparams["episodes"]):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.act(observation)

        if not action:
            action = 0

        if epsilon > np.random.random():
            action = np.random.randint(0, 2)
        
        wandb.log({"action": action})
        wandb.log({"epsilon": epsilon})
        print("Agent action", action)
        observation, reward, terminated, truncated, info = env.step(action)
        agent.assign_reward(reward)
        done = terminated or truncated

    epsilon = max(epsilon - 0.001, 0)

    episode_stats = {
        "episode": episode,
        "total_return": sum(agent.current_episode_rewards),
        "message_ct": len(agent.current_episode_messages),
        "episode_messages": agent.current_episode_messages,
    }
    train_stats = agent.terminate_episode()
    episode_stats.update(train_stats)
    wandb.log(episode_stats)