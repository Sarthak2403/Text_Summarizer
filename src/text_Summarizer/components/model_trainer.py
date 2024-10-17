# from transformers import TrainingArguments, Trainer
# from transformers import DataCollatorForSeq2Seq
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from datasets import load_dataset, load_from_disk
# import torch
# import os
# from text_Summarizer.entity import ModelTrainerConfig  

# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config


    
#     def train(self):
#         device = "mps" if torch.backends.mps.is_available() else "cpu"
#         # device = "cuda" if torch.cuda.is_available() else "cpu"
#         tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
#         model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
#         seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
#         #loading data 
#         dataset_samsum_pt = load_from_disk(self.config.data_path)

#         # trainer_args = TrainingArguments(
#         #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
#         #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
#         #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
#         #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
#         #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
#         # ) 


#         trainer_args = TrainingArguments(
#             output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
#             per_device_train_batch_size=1, per_device_eval_batch_size=1,
#             weight_decay=0.01, logging_steps=10,
#             evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
#             gradient_accumulation_steps=16
#         ) 

#         trainer = Trainer(model=model_pegasus, args=trainer_args,
#                   tokenizer=tokenizer, data_collator=seq2seq_data_collator,
#                   train_dataset=dataset_samsum_pt["test"], 
#                   eval_dataset=dataset_samsum_pt["validation"])
        
#         trainer.train()

#         ## Save model
#         model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
#         ## Save tokenizer
#         tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))

# from transformers import TrainingArguments, Trainer
# from transformers import DataCollatorForSeq2Seq
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from datasets import load_dataset, load_from_disk
# import torch
# import os
# from text_Summarizer.entity import ModelTrainerConfig  

# # Set high watermark ratio for MPS (for Apple Silicon devices)
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config

#     def train(self):
#         # Use MPS if available, otherwise CPU (can switch to CUDA if needed)
#         # device = "mps" if torch.backends.mps.is_available() else "cpu"
#         # device = "cpu"
#         device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
#         # Load tokenizer and model for BART
#         tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
#         model_bart = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
#         seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_bart)
        
#         # Load the processed dataset
#         dataset_samsum_pt = load_from_disk(self.config.data_path)

#         # Training arguments with small batch size and gradient accumulation for memory efficiency
#         trainer_args = TrainingArguments(
#             output_dir=self.config.root_dir,               # Directory for saving models
#             num_train_epochs=1,                            # Number of training epochs
#             warmup_steps=500,                              # Warmup steps for learning rate
#             per_device_train_batch_size=1,                 # Batch size per device (adjust if needed)
#             per_device_eval_batch_size=1,                  # Batch size for evaluation
#             weight_decay=0.01,                             # Weight decay to avoid overfitting
#             logging_steps=10,                              # Frequency of logging
#             evaluation_strategy='steps',                   # Evaluation after every `eval_steps`
#             eval_steps=500,                                # Perform evaluation every 500 steps
#             save_steps=1e6,                                # Save model every million steps (no frequent saves)
#             gradient_accumulation_steps=64,                # Accumulate gradients over 16 steps
#             # fp16= True                                

#         )

#         # Define the trainer with the loaded dataset
#         trainer = Trainer(
#             model=model_bart,                              # BART model
#             args=trainer_args,                             # Training arguments
#             tokenizer=tokenizer,                           # Tokenizer for BART
#             data_collator=seq2seq_data_collator,           # Data collator for seq2seq tasks
#             train_dataset=dataset_samsum_pt["train"],      # Use the correct train dataset
#             eval_dataset=dataset_samsum_pt["validation"]   # Use the validation dataset
#         )
        
#         # Train the model
#         trainer.train()

#         # Save the trained model and tokenizer
#         tokenizer.save_pretrained(os.path.join(self.config.root_dir, "bart-samsum-model"))
#         tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

# from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
# from datasets import load_from_disk
# import os
# import torch
# from pyspark import SparkContext, SparkConf
# from text_Summarizer.entity import ModelTrainerConfig

# # Set high watermark ratio for MPS (for Apple Silicon devices)
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# # Disable tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config
#         # Configure Spark
#         conf = SparkConf().setAppName("TextSummarization").setMaster("local[*]")
#         self.sc = SparkContext(conf=conf)

#     def train(self):
#         # Set the device for model training
#         device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#         # Load tokenizer and model
#         tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
#         model_bart = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
#         seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_bart)

#         # Load the dataset from disk
#         dataset_samsum_pt = load_from_disk(self.config.data_path)

#         # Distribute the training dataset using Spark RDD
#         train_dataset_rdd = self.sc.parallelize(dataset_samsum_pt["train"])

#         # Function to process each batch in parallel using Spark
#         def process_batch(batch):
#             return tokenizer(batch["dialogue"], padding="max_length", truncation=True, return_tensors="pt")

#         # Apply the processing function to the RDD
#         tokenized_train_rdd = train_dataset_rdd.map(process_batch)

#         # Collect the processed data back into a list (or can be used for in-memory operations)
#         tokenized_train_data = tokenized_train_rdd.collect()

#         # Prepare training arguments
#         trainer_args = TrainingArguments(
#             output_dir=self.config.root_dir,
#             num_train_epochs=1,
#             warmup_steps=500,
#             per_device_train_batch_size=1,
#             per_device_eval_batch_size=1,
#             weight_decay=0.01,
#             logging_steps=10,
#             evaluation_strategy="steps",
#             eval_steps=500,
#             save_steps=1e6,
#             gradient_accumulation_steps=64,
#         )

#         # Define Trainer
#         trainer = Trainer(
#             model=model_bart,
#             args=trainer_args,
#             tokenizer=tokenizer,
#             data_collator=seq2seq_data_collator,
#             train_dataset=dataset_samsum_pt["train"],
#             eval_dataset=dataset_samsum_pt["validation"]
#         )

#         # Train the model
#         trainer.train()

#         # Save the trained model and tokenizer
#         tokenizer.save_pretrained(os.path.join(self.config.root_dir, "bart-samsum-model"))
#         tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

#     def stop_spark(self):
#         # Stop the SparkContext
#         self.sc.stop()


from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import os
import torch
from text_Summarizer.entity import ModelTrainerConfig

# Set high watermark ratio for MPS (for Apple Silicon devices)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Set the device for model training
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_bart = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_bart)

        # Load the dataset from disk
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Tokenize the training dataset
        def tokenize_function(examples):
            return tokenizer(examples["dialogue"], padding="max_length", truncation=True)

        tokenized_dataset = dataset_samsum_pt.map(tokenize_function, batched=True)

        # Prepare training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            warmup_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1e6,
            gradient_accumulation_steps=64,
        )

        # Define Trainer
        trainer = Trainer(
            model=model_bart,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=tokenized_dataset["test"],
            eval_dataset=tokenized_dataset["validation"]
        )

        # Train the model
        trainer.train()

        # Save the trained model and tokenizer
        model_bart.save_pretrained(os.path.join(self.config.root_dir, "bart-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

if __name__ == "__main__":
    # Example usage
    config = ModelTrainerConfig(
        model_ckpt="facebook/bart-large-cnn",  # or your specific model checkpoint
        data_path="path/to/your/dataset",  # specify the path to your dataset
        root_dir="path/to/save/model"  # specify where to save the model
    )

    trainer = ModelTrainer(config)
    trainer.train()
