import numpy as np

def tokenize_worker(b):
    tmp = np.zeros(1000)
    tokenizer = b[2].from_pretrained("../data/features/bert/")
    tokens = tokenizer.encode_plus(b[0],padding=False,return_attention_mask=False,return_tensors='np')['input_ids']
    for tok in tokens:
        tmp[tok]+=1
    return tmp,b[1]

def tokenize_worker2(b):
    inp =  b[2].encode_plus(b[0],add_special_tokens = True,
                                max_length = 1500,        
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'pt', 
                                truncation = True)
    return inp,b[1]