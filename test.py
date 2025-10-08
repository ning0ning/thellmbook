from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",  # CUDA 대신 CPU 사용
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

#create a pipeline
generator = pipeline( 
       "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False
)

#the prompt(user input/query)
messages = [ 
    {"role":"user","content":"Create a funny joke about chickens."}
]

#generate output
output = generator(messages)
print(output[0]["generated_text"]) #