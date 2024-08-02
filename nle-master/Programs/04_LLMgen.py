import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel , prepare_model_for_kbit_training , get_peft_model
from trl import SFTTrainer
from huggingface_hub import login 


login(token="xxxxxxxxxxxxxxxxxxxxxxxx")

### first step ####

G


#### go to cmd => huggingface-cli login ##### 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_model_id = 'mistralai/Mixtral-8x7B-v0.1'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16, #if your gpu supports it 
    bnb_4bit_quant_type = "nf4",
    #bnb_4bit_use_double_quant = False #this quantises the quantised weights
)

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")



# Training_tokenizer (https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)
# https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    truncation_side = "right",
    padding_side="right",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


train_ds = load_dataset("json" , data_files = 'codes.jsonl' , field = "train")
test_ds = load_dataset("json" , data_files = 'codes.jsonl' , field = "test")

print(train_ds)


base_model.gradient_checkpointing_enable() #this to checkpoint grads 
model = prepare_model_for_kbit_training(base_model) #quantising the model (due to compute limits)








