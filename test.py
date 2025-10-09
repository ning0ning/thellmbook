from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",  # CUDA 대신 CPU 사용
    dtype="auto",
    # trust_remote_code=True,
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
#output[0]:리스트의 첫번째 결과 가져오기
#["generated_text"]: 그 결과안에서 "generated_text"에 해당하는 값 꺼내기
print(output[0]["generated_text"])

print(output)

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
print(input_ids)
print(generation_output)

#input_ids[0]:입력문장을 토크나이즈한 결과중 첫번째 문장을 가져옵니다.
#for id in input_ids:그 리스트안에 있는 각토큰(숫자)를 하나씩 꺼내서 반복
#tokenizer.decode(id): 숫자 토큰 하나를 사람이 읽을 수 있는 글자로 바꿈
'''for id in input_ids[0]:
    print(tokenizer.decode(id))'''

'''
print(tokenizer.decode(3233))
print(tokenizer.decode(622))
print(tokenizer.decode([3323,622]))
print(tokenizer.decode(29901))
'''

text = """
english and capitalization
show_tokens False None elif == >= else: two tabs:" " Three tabs:
12.0*50 = 600
"""

colors_list = [ 
   '102;194;165','252;141;98','141;160;203',
   '231;138;195','166;216;84','255;217;47'
]

def show_tokens(sentence,tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
    for idx, t in enumerate(token_ids):
        print( 
            f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' +
                tokenizer.decode(t) +'\x1b[0m',end='' 
             )