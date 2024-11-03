import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import json

class LlamaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False

class LlamaTrainer(LlamaModel):
    def __init__(self, model_name, data_path, prompt_path, examples_path):
        super().__init__(model_name)
        self.load_tokenizer()
        self.load_model()
        self.data_path = data_path
        self.prompt_path = prompt_path
        self.examples_path = examples_path

    def prepare_dataset(self):
        train_df = pd.read_csv(self.data_path + "/train.csv")
        val_df = pd.read_csv(self.data_path + "/validation.csv")
        
        with open(self.prompt_path, "r") as f:
            system_prompt = f.read()

        with open(self.examples_path, "r") as f:
            few_shot_examples = json.load(f)["few-shots"]

        def process_data(dataframe):
            texts = []
            for _, row in dataframe.iterrows():
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(few_shot_examples)
                messages.append({"role": "user", "content": row['text']})
                texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))
            
            return Dataset.from_dict({"text": texts})

        train_dataset = process_data(train_df)
        val_dataset = process_data(val_df)
        
        return train_dataset, val_dataset

    def train(self):
        train_dataset, val_dataset = self.prepare_dataset()

        peft_config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",],
        )

        training_arguments = TrainingArguments(
            output_dir="./model",
            optim="paged_adamw_32bit",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=2e-4,
            fp16=True,
            num_train_epochs=3,
            warmup_ratio=0.1,
            overwrite_output_dir=True,
            lr_scheduler_type="cosine",
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=64,
            tokenizer=self.tokenizer,
            args=training_arguments,
        )
        trainer.model.print_trainable_parameters()
        trainer.train()
        trainer.model.save_pretrained("./model")

class LlamaInference(LlamaModel):
    def __init__(self, model_name, prompt_path, examples_path, adapter_path=None):
        super().__init__(model_name)
        self.load_tokenizer()
        self.load_model()
        with open(prompt_path, "r") as f:
            self.system_prompt = f.read()
        with open(examples_path, "r") as f:
            self.few_shot_examples = json.load(f)["few-shots"]

        if adapter_path:
            self.load_adapter(adapter_path)

    def load_adapter(self, adapter_path):
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"Adapter loaded from {adapter_path}")

    def inference(self, input_text):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.few_shot_examples)
        messages.append({"role": "user", "content": input_text})
        
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=64, do_sample=False, top_p=None, temperature=None)
        
        # Extract only the new tokens generated by the model
        new_tokens = outputs[0][inputs.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--examples_path", required=True)
    parser.add_argument("--adapter_path", help="Path to the trained adapter for inference")
    args = parser.parse_args()

    if args.mode == "train":
        trainer = LlamaTrainer(args.model_name, args.data_path, args.prompt_path, args.examples_path)
        trainer.train()
    elif args.mode == "inference":
        inferencer = LlamaInference(args.model_name, args.prompt_path, args.examples_path, args.adapter_path)
        while True:
            input_text = input("Enter your question (or 'quit' to exit): ")
            if input_text.lower() == 'quit':
                break
            response = inferencer.inference(input_text)
            print("Response:", response)

if __name__ == "__main__":
    main()