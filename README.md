# Hugging Face - AI Agents course

This repository contains code and resources for the **Hugging Face AI Agents Course**. It demonstrates various concepts and implementations of AI agents, including tools, workflows, and frameworks like `smolagents`, `LangGraph`, and `LlamaIndex`. In this README I tried to include the most important theory behind AI agents, as briefly as possible.

## Repository Structure

    ├── gala_example/ # Example implementation of a gala planning agent 
    ├── img/ # Images used in documentation 
    ├── langgraph/ # LangGraph-based agent implementations 
    ├── smolagents/ # Smolagents-based agent implementations 
    └── Hugging Face - AI Agents course.md # Course notes and explanations

## AI agents
AI Agents consist of two parts: 
- Brain: AI model
    - Most common is LLM: GPT4, Llama, Gemini etc
    - VLM
- Body: Capabilities and tools
    - eg Image generation, web search, send email etc

The agent uses the AI model to:
- Understand natural language
- reason and plan
- interact with the environment

The agent then will make use of multiple tools, to perform an action, based on what the user asked.

### LLMs

LLMs are built on Transformers architecture.
There are 3 types of transformers:
- encoders
- decoders
- seq2seq

The most common LLMs right now are decoders.

LLM's objective is to predict the next token, given a sequence of previous tokens. Each LLM also has special tokens such as "end of sentence" (EOS).
LLMs decode until they reach EOS.

Decoders, give a representation to the sentence we wrote, in the form that they give possibilities for each next potential token. 
The most common decoding strategy is that the token with the highest probability wins. An alternative one is beam search.

A new addition to transformers was attention, which identified the most relevant words to predict the next token, and has proven to be extremely efficient.

LLMs use chat templates so that users interact with their interface. So then the adjust their special tokens according to what model is used.

#### Messages

- System messages: 
    - eg "You are a professional customer service agent. Always be polite, clear, and helpful."
- Conversations:
    - It is the conversation that user and LLM exchange. For every message, the LLM concatenate the entire conversation into a single prompt which inputs it into the model.

#### Chat templates

Chat templates are divided into base and instruct models. Base are trained on raw text, while instruct models use specific and different conversation formats and special tokens. Chat templates are implemented to ensure that we correctly format the prompt the way each model expects.

So to convert a conversation into a prompt, we load chat template from the tokenizer of the model.

### VLMs
Vision language models are broadly defined as multimodal models that can learn from images and text. The use cases include chatting about images, image recognition via instructions, visual question answering, document understanding, image captioning, and others. Some vision language models can also capture spatial properties in an image. 

One may find a VLM model based on the [OpenVLM leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard). 

<details>
<summary>VLM with transformers</summary>

#### VLM with transformers

We will go through how to use these models using transformers and fine-tune using SFTTrainer.
Let’s initialize the model and the processor first.

```
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to(device)
```

We now pass the image and the text prompt to the processor, and then pass the processed inputs to the generate. 
```
from PIL import Image
import requests

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
```

#### Fine-tuning Vision Language Models with TRL

```
from trl.commands.cli_utils import SftScriptArguments, TrlParser

parser = TrlParser((SftScriptArguments, TrainingArguments))
args, training_args = parser.parse_args_and_config()

LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""
```

We will now initialize our model and tokenizer.
```
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration
import torch

model_id = "llava-hf/llava-1.5-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer = tokenizer

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
```

Let’s create a data collator to combine text and image pairs.

```
class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

data_collator = LLavaDataCollator(processor)
```

Load our dataset.

```
from datasets import load_dataset

raw_datasets = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]
```

Initialize the SFTTrainer, passing in the model, the dataset splits, PEFT configuration and data collator and call train().
```
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",  # need a dummy field
    tokenizer=tokenizer,
    data_collator=data_collator,
    dataset_kwargs={"skip_prepare_dataset": True},
)

trainer.train()
```

The trained model can be found [here](https://huggingface.co/HuggingFaceH4/vsft-llava-1.5-7b-hf-trl).

</details>


#### VLM Models

- [Any-to-any models](https://huggingface.co/collections/merve/any-to-any-models-6822042ee8eb7fb5e38f9b62): Any-to-any models, as the name suggests, are models that can take in any modality and output any modality (image, text, audio). A capable such model is Qwen2.5-Omni.

- Reasoning models: They are models that can solve complex problems. An example is Kimi-VL-A3B-Thinking

- Smol yet Capable Models: When we say small vision language models we often refer to models with less than 2B parameters that can be run on consumer GPUs. SmolVLM is a good example model family for smaller vision language models. Another striking model is [gemma3-1b-it](https://huggingface.co/google/gemma-3-1b-it) by Google DeepMind. It’s particularly exciting as it’s one of the smallest multimodal models to have 32k token context window, and supports 140+ languages. An example of using these models:
```
python3 -m mlx_vlm.generate --model HuggingfaceTB/SmolVLM-500M-Instruct --max-tokens 400 --temp 0.0 --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vlm_example.jpg --prompt "What is in this image?"
```
```
llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF
```

- Mixture-of-Experts as Decoders: [Mixture of Expert (MoEs)](https://huggingface.co/blog/moe) models offer an alternative to dense architectures by dynamically selecting and activating only the most relevant sub-models, termed "experts", to process a given input data segment. MoEs need more memory cost due to the entire model being on the GPU. In the widely adopted Transformer architecture, MoE layers are most commonly integrated by replacing the standard Feed-Forward Network (FFN) layers within each Transformer block. Vision language models that have mixture-of-experts decoders seem to have enhanced performance.

- Vision-Language-Action Models: VLMs for robotics. VLAs take images and text instructions, and return text indicating actions for the robot to take directly. VLAs extend vision language models by adding action and state tokens to interact with and control physical environments. VLAs are usually fine-tuned on top of a base VLM. Examples are [π0 and π0-FAST](https://huggingface.co/lerobot/pi0) and [GR00T N1](https://huggingface.co/nvidia/GR00T-N1-2B)

#### Specialized Capabilities

Some capabilities are:
Object Detection, Segmentation, Counting with Vision Language Models.

An example is [PaliGemma](https://huggingface.co/blog/paligemma). For detection, the model outputs the bounding box coordinates as tokens. For segmentation, on the other hand, the model outputs detection tokens and segmentation tokens. These segmentation tokens aren’t all the segmented pixel coordinates, but codebook indices that are decoded by a variational autoencoder trained to decode these tokens into valid segmentation masks (as shown in the figure below).

Other models are PaliGemma 2, Molmo (for object counting), Qwen2.5-VL (detect, point to and count objects)

Multimodal Safety Models are also specialized capability. They are used before and after vision language models to filter their inputs and outputs. Examples are ShieldGemma 2, Llama Guard 4.

Another capability is: Multimodal RAG: retrievers, rerankers.
Multimodal retrievers take a stack of PDFs and a query as input and return the most relevant page numbers along with their confidence scores. The most relevant pages are then fed to the vision language model along with the query, and the VLM generates the answer.

#### Multimodal Agents

Vision language models unlock many agentic workflows from chatting with documents to computer use. This is possible by operating over GUIs.

For example, in order to describe documents (static):
```
agent = CodeAgent(tools=[], model=model) # no need for tools
agent.run("Describe these documents:", images=[document_1, document_2, document_3])
```

If we want the agent to take screenshots for browser/UI use (dynamic => callback):
```
def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    """ 
    Takes screenshots and writes to observations.
"""
  png_bytes = driver.get_screenshot_as_png()
        memory_step.observations_images = [image.copy()]  # persist images to memory_step
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return

agent = CodeAgent(
    tools=[go_back, close_popups, search_item_ctrl_f], # pass navigation tools
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot], # pass callback
)
```
or in a CLI command:
```
webagent "go to xyz.com/men, get to sale section, click the first clothing item you see. Get the product details, and the price, return them. note that I'm shopping from France"
```

#### Video Language Models

Most vision language models these days can handle videos, because videos can be represented as a sequence of frames. However, video understanding is tricky because of the temporal relationship between frames and the large amount of frames, so different techniques are used to select a representative set of video frames.
Different approaches to this problem are:
- [LongVU model](https://huggingface.co/collections/Vision-CAIR/longvu-67181d2debabfc1eb050c21d) 
- [Qwen2.5VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) 
- [Gemma 3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)


### Tools

A Tool is a function given to the LLM. This function should fulfill a clear objective.
eg web search, image generation, api interface etc.

A Tool should contain:
- A textual description of what the function does.
- A Callable (something to perform an action).
- Arguments with typings.
- (Optional) Outputs with typings.

When we write to LLMs, the LLM recognizes the need of tool, so it forms the prompt into a code text. Then the agent invokes the tool and returns the result back to LLM.

We give tools to the LLM in the system message.
We give it in a descriptive way such as:
"Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int"
Or, alternatively, we make use of python's auto-formatting. So we use Python's decorators to define a function as a tool.


## AI Agent Workflow

Thought-Action-Observation: Agents iterate through a loop until the objective is fullfilled. The agent starts with a thought, then acts by calling a tool, and finally observes the outcome. If the observation had indicated an error or incomplete data, the agent could have re-entered the cycle to correct its approach.

### Thought

- ReAct

ReAct is a simple prompting technique that appends “Let’s think step by step” before letting the LLM decode the next tokens. This allows the model to consider sub-steps in more detail, which in general leads to less errors than trying to generate the final solution directly.

### Actions

Actions are the concrete steps an AI agent takes to interact with its environment.
Examples:
- JSON
- Code
- Function-calling Agent

Actions are implemented as stop and parse approach: They generate the action in a known (eg JSON, or even better as a code) format, once the action is complete, they can be stopped, and the parser reads the formatted action.

eg (JSON)
```
Thought: I need to check the current weather for New York.
Action :
{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}
```

or with code:
```
# Code Agent Example: Retrieve Weather Information
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

# Execute the function and prepare the final answer
result = get_weather("New York")
final_answer = f"The current weather in New York is: {result}"
print(final_answer)
```

### Observations

Observations are how an Agent perceives the consequences of its actions.
- Collects Feedback
- Appends Results
- Adapts its Strategy

After performing an action, the framework follows these steps in order:

1. Parse the action to identify the function(s) to call and the argument(s) to use.
2. Execute the action.
3. Append the result as an Observation.

## AI Agent into practice

For a complete AI agent example, check out the smolagents/dummy_agent.py file.

<details>
<summary>Fine-tune a dataset</summary>

### Fine-tune a dataset

#### Preprocessing a dataset

To tokenize a whole dataset, we can feed the tokenizer a list of pairs of sentences by giving it the list of first sentences, then the list of second sentences. So, one way to preprocess the training dataset is:
```
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```
But this returns a dictionary, and takes a lot of RAM space (as opposed to datasets that are saved as Apache Arrow files). To keep it as a dataset, we use the Dataset.map() method, eg:
```
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

#### Padding

If we use TPU, then we could make use of PyTorch tensors with a collate function. This needs same size rows, so we would use padding.
```
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

so in total:
```
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

#### Train

```
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Define the trainer that will help us to fine-tune. We will set evaluation_strategy, epoch and compute_metrics(), from the Evaluate library, to evaluate the training:
```
from transformers import Trainer

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

To get the predictions:
```
predictions = trainer.predict(tokenized_datasets["validation"]) #it returns logits
print(predictions.predictions.shape, predictions.label_ids.shape)

# take the index with the maximum value on the second axis:
preds = np.argmax(predictions.predictions, axis=-1)
```

To evaluate:
```
import evaluate
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

Full training, without using Trainer. For it we also need an optimizer, a learning rate, and a device to do the train on a GPU:
```
# some preprocessing
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# dataloader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# optimizer
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

# learning rate. For it we need the number of training steps
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# device
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
device(type='cuda')

# add a progress bar to know how long it will take:
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

# train
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# evaluate
import evaluate
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
metric.compute()

```

To inspect a batch:
```
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
```

We can also use the Accelerate library in order to train on multiple GPUs or TPUs (examples can be found in https://github.com/huggingface/accelerate/tree/main/examples):
```
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# to try it out:
accelerate config

# or to launch it:
accelerate launch train.py

# to try it in a Notebook (eg Collab):
from accelerate import notebook_launcher
notebook_launcher(training_function)
```
</details>

<details>
<summary>Supervised fine-tune a dataset</summary>

### Supervised fine-tune a dataset

The previous method, fine-tunes the dataset for a specific task. To fine-tune it on a broad range of tasks simultaneously, we use supervised fine-tuning (SFT). This turns them more capable assistant models.
SFT involves significant computational resources and engineering effort, so it should only be pursued when prompting existing models proves insufficient.

Consider SFT only if you: - Need additional performance beyond what prompting can achieve - Have a specific use case where the cost of using a large general-purpose model outweighs the cost of fine-tuning a smaller model - Require specialized output formats or domain-specific knowledge that existing models struggle with.

#### Dataset preparation

- Training parameters
    - Training Duration
        - num_train_epochs
        - max_steps (more epochs, better learning but risk overfitting)
    - batch size
        - per_device_train_batch_size
        - gradient_accumulation_steps (Larger batches provide more stable gradients but require more memory)
    - learning rate
        - learning_rate
        - warmup_ratio (Too high can cause instability, too low results in slow learning)
    - monitoring
        - logging_steps
        - eval_steps
        - save_steps

Implementation with Transformers Reinforcement Learning (TRL):
```
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

# Configure trainer
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

# Start training
trainer.train()
```

Packing the dataset maximizes GPU utilization during training. We can define a custom formatting function to combine the fields into a single input sequence:
```
def formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


training_args = SFTConfig(packing=True)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)
```

#### Monitoring

Watch for these warning signs during training: 1. Validation loss increasing while training loss decreases (overfitting) 2. No significant improvement in loss values (underfitting) 3. Extremely low loss values (potential memorization) 4. Inconsistent output formatting (template learning issues). Ideal training shows a small gap between training and validation loss, suggesting the model is learning generalizable patterns.

If there is overfitting, consider:
- Reducing the training steps
- Increasing the dataset size
- Validating dataset quality and diversity

Extremely low loss values could suggest memorization rather than learning. 

#### [LoRA](https://huggingface.co/learn/llm-course/en/chapter11/4?fw=pt})

LoRA is a technique that allows us to fine-tune large language models with a small number of parameters. It works by adding and optimizing smaller matrices to the attention weights, typically reducing trainable parameters by about 90%.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into the model’s layers. Instead of training all model parameters during fine-tuning, LoRA decomposes the weight updates into smaller matrices through low-rank decomposition, significantly reducing the number of trainable parameters while maintaining model performance. For example, when applied to GPT-3 175B, LoRA reduced trainable parameters by 10,000x and GPU memory requirements by 3x compared to full fine-tuning. You can read more about LoRA in the LoRA paper.

LoRA works by adding pairs of rank decomposition matrices to transformer layers, typically focusing on attention weights. During inference, these adapter weights can be merged with the base model, resulting in no additional latency overhead. LoRA is particularly useful for adapting large language models to specific tasks or domains while keeping resource requirements manageable.

![LoRA](img/blog_multi-lora-serving_LoRA.gif "LoRA Example")

To load LoRA with PEFT library:
```
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
```

And combined with the TRL library:
```
from peft import LoraConfig

# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 6
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 8
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)

# Create SFTTrainer with LoRA configuration
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,  # LoRA configuration
    max_seq_length=max_seq_length,  # Maximum sequence length
    processing_class=tokenizer,
)
```

After training a LoRA adapter, you can merge the adapter weights back into the base model.

```
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name", torch_dtype=torch.float16, device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model, "path/to/adapter", torch_dtype=torch.float16
)

# 3. Merge adapter weights with base model
merged_model = peft_model.merge_and_unload()
```

#### Evaluation

Some automatic benchmarks:
- General Knowledge Benchmarks like [MMLU](https://huggingface.co/datasets/cais/mmlu) (Massive Multitask Language Understanding)
- Reasoning Benchmarks like [BBH](https://huggingface.co/datasets/lukaemon/bbh) (Big Bench Hard) and [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- Language Understanding like [HELM](https://github.com/stanford-crfm/helm)
- Domain-Specific Benchmarks like [MATH](https://huggingface.co/papers/2103.03874), [HumanEval](https://github.com/openai/human-eval), [Alpaca](https://tatsu-lab.github.io/alpaca_eval/)
- Alternatives like LLM-as-Judge, Evaluation Arenas like [Chatbot Arena](https://lmarena.ai/)
- Custom evaluation: Use [lighteval](https://github.com/huggingface/lighteval)

Here’s a complete example of evaluating on automatic benchmarks relevant to one specific domain using Lighteval with the VLLM backend:
```
lighteval accelerate \
    "pretrained=your-model-name" \
    "mmlu|anatomy|0|0" \
    "mmlu|high_school_biology|0|0" \
    "mmlu|high_school_chemistry|0|0" \
    "mmlu|professional_medicine|0|0" \
    --max_samples 40 \
    --batch_size 1 \
    --output_path "./results" \
    --save_generations true
```
</details>

### Function calling

[Function-calling](https://docs.mistral.ai/capabilities/function_calling/) is a way for an LLM to take actions on its environment. The agent is fine-tuned to use Tools instead of generalizing on defining a plan using these Tools.

Function-calling brings new roles to the conversation!
- One new role for an Action
- One new role for an Observation

To make use of function calling, follow [these guidelines](https://huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb):

1. Install dependencies
2. Import libraries
3. Obtain Anthropic or OpenAI API token and add in the code
4. Preprocess the input into what we want the model to learn, by using a chat template
5. Give a subset of the dataset to compute some thinking tokens, before the function call
6. Configure LoRA
7. Define Trainer and fine-tuning parameters
8. Train the model and save it.
9. Push the model and tokenizer to the hub.
10. Test the model

## Frameworks for AI Agents

### smolagents

In short, smolagents is a library that focuses on codeAgent, a kind of agent that performs “Actions” through code blocks, and then “Observes” results by executing the code.

smolagents use CodeAgent as the primary type of agent, which outputs Python code, but also supports ToolCallingAgent, which writes tool calls in JSON.

smolagents provides several predefined classes to simplify model connections:

- TransformersModel: Implements a local transformers pipeline for seamless integration.
- HfApiModel: Supports serverless inference calls through Hugging Face’s infrastructure, or via a growing number of third-party inference providers.
- LiteLLMModel: Leverages LiteLLM for lightweight model interactions.
- OpenAIServerModel: Connects to any service that offers an OpenAI API interface.
- AzureOpenAIServerModel: Supports integration with any Azure OpenAI deployment.

* The advantage of smolagents is that it uses a code-first (Python) approach, compared to the JSON/text approach of others.

<details>
<summary>For inspection and logging, smolagents use OpenTelemetry</summary>

OpenTelemetry:

```
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
```

To easily track and analyze the agent's behavior via Langfuse and SmolagentsInstrumentor:

```
import os
import base64

LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # US data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"


from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
```

It can then access and analyze the logs [here](https://cloud.langfuse.com/project/cm7bq0abj025rad078ak3luwi/traces/995fc019255528e4f48cf6770b0ce27b?timestamp=2025-02-19T10%3A28%3A36.929Z)

</details>


smolagents' tools need an interface description with these key components:

- Name: What the tool is called
- Tool description: What the tool does
- Input types and descriptions: What arguments the tool accepts
- Output type: What the tool returns

and can be defined:
- Using the @tool decorator for simple function-based tools
- Creating a subclass of Tool for more complex functionality

Default toolbox:
- PythonInterpreterTool
- FinalAnswerTool
- UserInputTool
- DuckDuckGoSearchTool
- GoogleSearchTool
- VisitWebpageTool

#### Multi-agent systems

A typical setup might include:

- A Manager Agent for task delegation
- A Code Interpreter Agent for code execution
- A Web Search Agent for information retrieval

eg a Multi-Agent RAG system integrates:
- Orchestrator agent
    - A Web Agent
    - A Retriever Agent
    - An Image Generation Agent


<details>
<summary>Llamaindex</summary>

### Llamaindex

The main parts of Llamaindex are Components, Agents and Tools and Workflows:
- Components: Are the basic building blocks you use in LlamaIndex. These include things like prompts, models, and databases. Components often help connect LlamaIndex with other tools and libraries.
- Tools: Tools are components that provide specific capabilities like searching, calculating, or accessing external services. They are the building blocks that enable agents to perform tasks.
- Agents: Agents are autonomous components that can use tools and make decisions. They coordinate tool usage to accomplish complex goals.
- Workflows: Are step-by-step processes that process logic together. Workflows or agentic workflows are a way to structure agentic behaviour without the explicit use of agents.

#### [Components](https://docs.llamaindex.ai/en/stable/module_guides/)

An example of a component is a [RAG pipeline](https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/components.ipynb). QueryEngine is an example of RAG, a component that searches through your relevant information to be helpful.

* What is [RAG](https://docs.llamaindex.ai/en/stable/understanding/rag/): LLMs are trained on enormous bodies of data to learn general knowledge. However, they may not be trained on relevant and up-to-date data. RAG solves this problem by finding and retrieving relevant information from your data and giving that to the LLM.

The key stages of a RAG are:

1. Loading and embedding documents:
* SimpleDirectoryReader
* [LlamaParse](https://github.com/run-llama/llama_cloud_services/blob/main/parse.md)
* [LlamaHub](https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/)

After loading our documents, we need to break them into smaller pieces called Node objects.
The IngestionPipeline helps us create these nodes through SentenceSplitter and HuggingFaceEmbedding 

2. Storing and indexing

A key feature of RAG is indexing: creation of data structure (vector embeddings) out of the data to make it easy to accurately find contextually relevant data based on properties.

```
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

nodes = await pipeline.arun(documents=[Document.example()])
```

VectorStoreIndex helps us find relevant matches by embedding both the query and nodes in the same vector space:
```
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
```

3. Querying a VectorStoreIndex with prompts and LLMs

To convert the index to a query interface:
* as_retriever
* as_query_engine
* as_chat_engine

```
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)
query_engine.query("What is the meaning of life?")
# The meaning of life is 42
```

4. Response Processing

The query engine uses ResponseSynthesizer as a strategy to process the response. The three main strategies for it are:
* refine
* compact 
* tree_summarize

To customize and fine-tune every step of the query process to match your exact needs use the [low-level composition API](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/usage_pattern/#low-level-composition-api)

5. Evaluation and observability

LlamaIndex provides built-in evaluation tools to assess response quality:
* FaithfulnessEvaluator
* AnswerRelevancyEvaluator
* CorrectnessEvaluator

```
from llama_index.core.evaluation import FaithfulnessEvaluator

query_engine = # from the previous section
llm = # from the previous section

# query index
evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response)
eval_result.passing
```

Even without direct evaluation, we can gain insights into how our system is performing through observability:
```
import llama_index
import os

PHOENIX_API_KEY = "<PHOENIX_API_KEY>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix",
    endpoint="https://llamatrace.com/v1/traces"
)
```

#### Tools

There are four main types of tools in LlamaIndex:

1. [FunctionTool](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/modules/function_calling.html)

```
from llama_index.core.tools import FunctionTool

def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)
tool.call("New York")
```

2. QueryEngineTool

```
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(llm=llm)
tool = QueryEngineTool.from_defaults(query_engine, name="some useful name", description="some useful description")
```

3. Toolspecs

```
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

[(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]
```

4. Model Context Protocol (MCP)

```
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# We consider there is a mcp server running on 127.0.0.1:8000, or you can use the mcp client to connect to your own mcp server.
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool = McpToolSpec(client=mcp_client)

# get the agent
agent = await get_agent(mcp_tool)

# create the agent context
agent_context = Context(agent)
```

4. Utility Tools

Oftentimes, directly querying an API can return an excessive amount of data, some of which may be irrelevant, overflow the context window of the LLM, or unnecessarily increase the number of tokens that you are using. Two utility tools can help with this:

* OnDemandToolLoader
* LoadAndSearchToolSpec

#### Agents in LlamaIndex

LlamaIndex supports three main types of reasoning agents:
* Function Calling Agents
* ReAct Agents
* [Advanced Custom Agents](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/agent/workflow/base_agent.py)

The steps are:
1. Initialise. 

To create an agent, we start by providing it with a set of functions/tools that define its capabilities.
Agents are stateless by default. Context objects adds remembering.

```
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)

# stateless
response = await agent.run("What is 2 times 2?")

# remembering state
from llama_index.core.workflow import Context

ctx = Context(agent)

response = await agent.run("My name is Bob.", ctx=ctx)
response = await agent.run("What was my name again?", ctx=ctx)
```

2. Creating RAG Agents with QueryEngineTools

Agentic RAG is a powerful way to use agents to answer questions about your data.

```
from llama_index.core.tools import QueryEngineTool

query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)
```

3. Creating Multi-agent systems

It can be done with [AgentWorkflow](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/). [Agents](https://docs.llamaindex.ai/en/stable/understanding/agent/) in LlamaIndex can also directly be used as tools for other agents, for more complex and custom scenarios.

```
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

# Run the system
response = await agent.run(user_msg="Can you add 5 and 3?")
```

#### Workflows

A workflow in LlamaIndex provides a structured way to organize your code into sequential and manageable steps. Such a workflow is created by defining Steps which are triggered by Events, and themselves emit Events to trigger further steps.

1. Creating Workflows

It consists of:
* Basic Workflow Creation
* Connecting Multiple Steps
* Loops and Branches
* We can draw the workflows
* State Management

By using type hinting the the union operator | we can create loops where LoopEvent is taken as input for the step and can also be returned as output.


```
from llama_index.core.workflow import Event
import random


class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


w = MultiStepWorkflow(verbose=False)
result = await w.run()
result

# Draw
from llama_index.utils.workflow import draw_all_possible_flows

w = ... # as defined in the previous section
draw_all_possible_flows(w, "flow.html")
```

State management is useful when you want to keep track of the state of the workflow, so that every step has access to the same state. We can do this by using the Context type hint on top of a parameter in the step function:

```
from llama_index.core.workflow import Context, StartEvent, StopEvent


@step
async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
    # store query in the context
    await ctx.set("query", "What is the capital of France?")

    # do something with context and event
    val = ...

    # retrieve query from the context
    query = await ctx.get("query")

    return StopEvent(result=val)
```

Alternatively, instead of manual workflow creation, we can use the AgentWorkflow class to create a multi-agent workflow. One agent must be designated as the root agent in the AgentWorkflow constructor.
Before starting the workflow, we can provide an initial state dict that will be available to all agents (eg  inject a counter to count function calls).

```
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.workflow import Context

# Define some tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # update our count
    cur_state = await ctx.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)

    return a + b

async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # update our count
    cur_state = await ctx.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)

    return a * b

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent"
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)

# run the workflow with context
ctx = Context(workflow)
response = await workflow.run(user_msg="Can you add 5 and 3?", ctx=ctx)

# pull out and inspect the state
state = await ctx.get("state")
print(state["num_fn_calls"])
```
</details>

<details>
<summary>LangGraph</summary>

### LangGraph

[LangGraph](https://langchain-ai.github.io/langgraph/) is a framework that allows you to build production-ready applications by giving you control tools over the flow of your agent. ([course](https://academy.langchain.com/courses/intro-to-langgraph)).

Use smolagents' code agents if you want freedom, LangGraph is you want control.

Note: If there are tables or images involved in a document, better first turn this into text, as LLMs understand text the best.

LangGraph is perhaps the most production-ready agent framework on the market.

LangGraph consists of:
* Nodes: represent individual processing steps (like calling an LLM, using a tool, or making a decision). Python functions
* Edges: define the possible transitions between steps. Connect nodes (if... then node1 else node2)
* State: is user defined and maintained and passed between nodes during execution. When deciding which node to target next, this is the current state that we look at. Python classes
* StateGraph: the container that holds your entire agent workflow:
```
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# Visualize it
display(Image(graph.get_graph().draw_mermaid_png()))

# or invoke
graph.invoke({"graph_state" : "Hi, this is Lance."})
```
* Langfuse: to observe and inspect the agent

For an example, check the email_processing.py and document_analysis_agent.py.
The first reads the received emails, classifies them according to whether it believes they are spam or not, and notifies the user.
The second one process images documents, extract text using vision models (Vision Language Model), performs calculations when needed (to demonstrate normal tools), analyzes content and provides concise summaries and executes specific instructions related to documents

</details>

## Observability and Evaluation

For implementation check [this notebook](https://huggingface.co/agents-course/notebooks/blob/main/bonus-unit2/monitoring-and-evaluating-agents-notebook.ipynb).
### Observability

Common observability tools for AI agents include platforms like [Langfuse](https://langfuse.com/) and [Arize](https://arize.com/). These tools help collect detailed traces and offer dashboards to monitor metrics in real-time, making it easy to detect problems and optimize performance.

Observability tools usually represent agent runs as traces and spans.

* Traces represent a complete agent task from start to finish (like handling a user query).
* Spans are individual steps within the trace (like calling a language model or retrieving data).

Key metrics that these tools monitor:
* Latency
* Costs: AI agents rely on LLM calls billed per token or external APIs. Frequent tool usage or multiple prompts can rapidly increase costs.
* Request Errors
* User Feedback (explicit, eg rating)
* Implicit User Feedback (eg repeated queries)
* Accuracy
* Automated Evaluation Metrics

### Evaluation

There are two categories of evaluations for AI agents: online evaluation and offline evaluation. Both are valuable, and they complement each other. We usually begin with offline evaluation, as this is the minimum necessary step before deploying any agent.

* Offline Evaluation: This involves evaluating the agent in a controlled setting, typically using test datasets, not live user queries. Eg for a math agent, the evaluation could be a dataset of 100 problems with known answers. The benefit is that it’s repeatable and you can get clear accuracy metrics since you have ground truth. 

* Online Evaluation: This refers to evaluating the agent in a live, real-world environment, i.e. during actual usage in production (eg success rates, user satisfaction scores, or other metrics). The advantage of online evaluation is that it captures things you might not anticipate in a lab setting. It provides a true picture of how the agent behaves in the wild. It consists of implicit and explicit user feedback, A/B tests

The best practice is a combination of the two: offline evaluation → deploy new agent version → monitor online metrics and collect new failure examples → add those examples to offline test set → iterate.

## Gala Example

The [example](https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction) of hugging face online course can be found in the gala_example folder.

## AI Agent for the GAIA benchmark

As part of the final step of the course, we had to build an AI agent and evaluate it based on the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA). GAIA is made of more than 450 non-trivial question with an unambiguous answer. and a score over 30% is considered as a competent AI agent.

### Inspirations

Some examples of powerful agents and their corresponding agents are:
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher):
    - Utilize a planner that generates a research question.
    - Utilize execution agents that gather relevant information.
    - Utilize a publisher that aggregates all findings into a comprehensive report. 

- [STORM](https://arxiv.org/abs/2402.14207) paper, the research team can be comprised of 8 agents and replicated [here](https://github.com/assafelovic/gpt-researcher/tree/master/multi_agents):
    - Human - The human in the loop that oversees the process and provides feedback to the agents.
    - Chief Editor - Oversees the research process and manages the team. This is the "master" agent that coordinates the other agents using Langgraph.
    - Researcher (gpt-researcher) - A specialized autonomous agent that conducts in depth research on a given topic.
    - Editor - Responsible for planning the research outline and structure.
    - Reviewer - Validates the correctness of the research results given a set of criteria.
    - Revisor - Revises the research results based on the feedback from the reviewer.
    - Writer - Responsible for compiling and writing the final report.
    - Publisher - Responsible for publishing the final report in various formats.

These follow these steps:
- Planning stage
- Data collection and analysis
- Review and revision
- Writing and submission
- Publication

- [Magentic One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)
    - Orchestrator:  The lead agent responsible for task decomposition, planning, directing other agents in executing subtasks, tracking overall progress, and taking corrective actions as needed
    - Other Agents:
        - Coder: Write code and reason to solve tasks
        - ComputerTerminal: Execute code written by the coder agent
        - Websurfer: Browse the internet
        - Filesurfer: Navigate files

- [Jina AI](https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/)
Loop in a [Search - Read - Reason - Search] sceme until you have an answer or until budget is exceeded.

### Build the agent

Based on the [open Deep Research](https://huggingface.co/blog/open-deep-research), a first step to build a more competent AI agent is to use CodeAgent, as [code is specifically designed to express complex sequences of actions](https://huggingface.co/papers/2402.01030). Also using smolagents, means better handling of the state of each step of the agent.

A next step is to provide it with the right tools. According to existing agents, a good set of tools needed are:
- A web browser agent: eg OpenAI's Operator. Alternatively, it can be a vision-based web browser like [this one](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py)
- A document reader/summarizer agent: a text inspector that can read any kind of text. An example is [this one](https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/scripts/text_inspector_tool.py)
- Data analysis / calculator agent: Manage numerical computations or statistical inference tasks.

Considering the division of a human brain into two systems: a fast, intuitive, unconscious part and a slow, analytical, conscious part, I decided on top of these tools, to use the following agents:
- Input and query parsing agent: Parse and comprehend the user's query, identifying key objectives, relevant keywords and constraints. It convert the raw text into a more structured data format for the subsequent layers.
- Orchestrator: In order to not fall into an infinite loop, the paper [Plan and Solve](https://arxiv.org/abs/2305.04091) suggests that we first devise a plan to divide the entire task into smaller subtasks and then carrying out the subtasks according to the plan. The orchestrator then acts as the central executive function, evaluating the overall task, decomposing it into manageable, deterministic, finite set of subtasks, and assigning these tasks to specialized agents. Key functions of the orchestrator are:
    - Task Decomposition: Break down complex queries into specific components (like “search the web,” “read documents,” “compute statistics”).
    - Agent Scheduling: Decide which sub-agent (or tool) works on which aspect, ensuring efficient and non-redundant processing.
    - Integration: Gather and review the outputs from the sub-agents, ensuring consistency and coherence in the final response
- Integration and synthesis agent: Act as the “final thought” process—integrating outputs from all specialized agents and rechecking for coherence and accuracy.
- Memory and context agent: Maintain short-term and long-term contextual information across interactions.
- Feedback loop and adaptive learning agent: Continually assess the performance of different modules, adjust strategies, and refine internal parameters when inconsistencies arise