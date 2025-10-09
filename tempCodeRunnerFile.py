prompt = "Write an email apologizing to Sarah for the tragic gardening mishap.Explain how it happened.<|assistant|>"

#Tokenize the input prompt
input_ids = tokenizer(prompt,return_tensors ='pt').input_ids.to("cpu")

#Generate the text
generation_output = model.generate( 
    input_ids=input_ids,
    max_new_tokens=20
)
#print the output
print(tokenizer.decode(generation_output[0]))

