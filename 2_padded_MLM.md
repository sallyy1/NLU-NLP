```python
##from transformers import BertTokenizer, BertForMaskedLM
import torch
```

    /home/hyunkyung_lee/.local/lib/python3.6/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AdamW

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")#, use_fast=False)
model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")
```

    Some weights of the model checkpoint at klue/bert-base were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'cls.seq_relationship.bias']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



```python
###model = AutoModelForMaskedLM.from_pretrained('spanbert-base-cased')


# model_name = "SpanBERT/spanbert-base-cased" 


# # Download pytorch model
# model = AutoModel.from_pretrained(model_name)
```


```python
corpus = []
corpus_mun = []
corpus_ban = []

with open('train_sample.tsv', 'r') as fp:
    text = fp.read().split('\n')
    
for line in text:
    sentence = line.split('\t')[0]
    corpus.append(sentence)
```


```python
text[196326]
```




    '2016년에 갤럭시 S7 에지가 폭발한 사건은 어느 지역에서 일어났는가?\t0'




```python
text[196327]
```




    '중국에서 아파트에서 추락하던 3세 아이를 살리고 자신은 혼수상태에 빠진 사람은 누구야?\t1'




```python
text[196328]
```




    '천중핑씨가 추락하는 아이를 구하고 뇌출혈로 인한 의식불명 상태에 빠진 건 언제야?\t1'




```python
corpus_mun = corpus[:196327]
corpus_ban = corpus[196327:]
```


```python
print(len(corpus_mun))
print(len(corpus_ban))
```

    196327
    212423



```python
# 1번 코퍼스 학습 (1. 문어체 수행)

inputs = tokenizer(corpus_mun, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
```


```python
inputs
```




    {'input_ids': tensor([[   2, 3671, 2145,  ...,    0,    0,    0],
            [   2,   11, 3854,  ...,    0,    0,    0],
            [   2, 3686, 6431,  ...,    0,    0,    0],
            ...,
            [   2, 5217, 2440,  ...,    0,    0,    0],
            [   2, 7275, 6551,  ...,    0,    0,    0],
            [   2, 5217, 2440,  ...,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            ...,
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0]])}




```python
inputs['labels'] = inputs.input_ids.detach().clone()
```


```python
inputs
```




    {'input_ids': tensor([[   2, 3671, 2145,  ...,    0,    0,    0],
            [   2,   11, 3854,  ...,    0,    0,    0],
            [   2, 3686, 6431,  ...,    0,    0,    0],
            ...,
            [   2, 5217, 2440,  ...,    0,    0,    0],
            [   2, 7275, 6551,  ...,    0,    0,    0],
            [   2, 5217, 2440,  ...,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            ...,
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0]]), 'labels': tensor([[   2, 3671, 2145,  ...,    0,    0,    0],
            [   2,   11, 3854,  ...,    0,    0,    0],
            [   2, 3686, 6431,  ...,    0,    0,    0],
            ...,
            [   2, 5217, 2440,  ...,    0,    0,    0],
            [   2, 7275, 6551,  ...,    0,    0,    0],
            [   2, 5217, 2440,  ...,    0,    0,    0]])}




```python
rand = torch.rand(inputs.input_ids.shape)
rand.shape
```




    torch.Size([196327, 128])




```python
rand
```




    tensor([[0.3443, 0.6897, 0.3441,  ..., 0.4138, 0.1326, 0.9039],
            [0.9687, 0.4886, 0.8729,  ..., 0.4838, 0.4935, 0.9391],
            [0.2604, 0.6478, 0.6461,  ..., 0.0158, 0.2135, 0.1544],
            ...,
            [0.0275, 0.4115, 0.6908,  ..., 0.6365, 0.7793, 0.4601],
            [0.2756, 0.9895, 0.1738,  ..., 0.2016, 0.9166, 0.8593],
            [0.8565, 0.1026, 0.5104,  ..., 0.2151, 0.7818, 0.8223]])




```python
mask_arr = rand < 0 #0.15
mask_arr
```




    tensor([[False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            ...,
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False]])




```python
#inputs.input_ids != 2
```




    tensor([[False,  True,  True,  ...,  True,  True,  True],
            [False,  True,  True,  ...,  True,  True,  True],
            [False,  True,  True,  ...,  True,  True,  True],
            ...,
            [False,  True,  True,  ...,  True,  True,  True],
            [False,  True,  True,  ...,  True,  True,  True],
            [False,  True,  True,  ...,  True,  True,  True]])




```python
#tokenizer
```




    PreTrainedTokenizerFast(name_or_path='klue/bert-base', vocab_size=32000, model_max_len=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})



# - MASKING 방법에 변화 주기


```python
corpus_mun[0]
```




    "서울과 충북 괴산에서 '국제 청소년포럼'을 여는 곳은?"




```python
tokenizer.encode(corpus_mun[0])
```




    [2,
     3671,
     2145,
     7249,
     25859,
     27135,
     11,
     3854,
     4857,
     2208,
     2731,
     11,
     1498,
     1428,
     2259,
     601,
     2073,
     35,
     3]




```python
tokenizer.convert_ids_to_tokens(encoded)
```




    ['[CLS]',
     '서울',
     '##과',
     '충북',
     '괴산',
     '##에서',
     "'",
     '국제',
     '청소년',
     '##포',
     '##럼',
     "'",
     '을',
     '여',
     '##는',
     '곳',
     '##은',
     '?',
     '[SEP]']




```python
tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded))
```




    "[CLS] 서울과 충북 괴산에서'국제 청소년포럼'을 여는 곳은? [SEP]"




```python
tokenizer.convert_tokens_to_ids(tokenizer.convert_ids_to_tokens(encoded))
```




    [2,
     3671,
     2145,
     7249,
     25859,
     27135,
     11,
     3854,
     4857,
     2208,
     2731,
     11,
     1498,
     1428,
     2259,
     601,
     2073,
     35,
     3]




```python
encoded = tokenizer.encode(corpus_mun[0]) # str() 타입의 문장을 인풋
print('*** 토큰화 결과 :',tokenizer.convert_ids_to_tokens(encoded), end='\n\n')
print('*** <문장으로 보기> :', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded)), end='\n\n')
print('*** 정수 인코딩 :',encoded, end='\n\n') # encoded.ids
print('*** 디코딩 :',tokenizer.decode(encoded), end='\n\n') # encoded.ids
```

    *** 토큰화 결과 : ['[CLS]', '서울', '##과', '충북', '괴산', '##에서', "'", '국제', '청소년', '##포', '##럼', "'", '을', '여', '##는', '곳', '##은', '?', '[SEP]']
    
    *** <문장으로 보기> : [CLS] 서울과 충북 괴산에서'국제 청소년포럼'을 여는 곳은? [SEP]
    
    *** 정수 인코딩 : [2, 3671, 2145, 7249, 25859, 27135, 11, 3854, 4857, 2208, 2731, 11, 1498, 1428, 2259, 601, 2073, 35, 3]
    
    *** 디코딩 : [CLS] 서울과 충북 괴산에서'국제 청소년포럼'을 여는 곳은? [SEP]
    



```python
inputs.input_ids[0]
```




    tensor([    2,  3671,  2145,  7249, 25859, 27135,    11,  3854,  4857,  2208,
             2731,    11,  1498,  1428,  2259,   601,  2073,    35,     3,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0])




```python
print(len(tokenizer.encode(corpus_mun[0]))) # 총 19개 토큰
print(len(inputs.input_ids[0])) # 전체 길이는 max_len = 128, 그중 패딩이 아닌 토큰은 총 19개
```

    19
    128



```python
encoded = tokenizer.encode('티베로의 가장 큰 장점은 무엇인가?') # str() 타입의 문장을 인풋
print('*** 토큰화 결과 :',tokenizer.convert_ids_to_tokens(encoded), end='\n\n')
print('*** <문장으로 보기> :', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded)), end='\n\n')
print('*** 정수 인코딩 :',encoded, end='\n\n') # encoded.ids
print('*** 디코딩 :',tokenizer.decode(encoded), end='\n\n') # encoded.ids
```

    *** 토큰화 결과 : ['[CLS]', '티', '##베', '##로', '##의', '가장', '큰', '장점', '##은', '무엇', '##인', '##가', '?', '[SEP]']
    
    *** <문장으로 보기> : [CLS] 티베로의 가장 큰 장점은 무엇인가? [SEP]
    
    *** 정수 인코딩 : [2, 1819, 2472, 2200, 2079, 3676, 1751, 5472, 2073, 3890, 2179, 2116, 35, 3]
    
    *** 디코딩 : [CLS] 티베로의 가장 큰 장점은 무엇인가? [SEP]
    



```python
print(len(tokenizer.encode('티베로의 가장 큰 장점은 무엇인가?'))) # 총 14개 토큰
###print(len(inputs.input_ids[0])) # 전체 길이는 max_len = 128, 그중 패딩이 아닌 토큰은 총 14개
```

    14



```python
encoded = tokenizer.encode('티베로의 가장 큰 장점은 뭐야?') # str() 타입의 문장을 인풋
print('*** 토큰화 결과 :',tokenizer.convert_ids_to_tokens(encoded), end='\n\n')
print('*** <문장으로 보기> :', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded)), end='\n\n')
print('*** 정수 인코딩 :',encoded, end='\n\n') # encoded.ids
print('*** 디코딩 :',tokenizer.decode(encoded), end='\n\n') # encoded.ids
```

    *** 토큰화 결과 : ['[CLS]', '티', '##베', '##로', '##의', '가장', '큰', '장점', '##은', '뭐', '##야', '?', '[SEP]']
    
    *** <문장으로 보기> : [CLS] 티베로의 가장 큰 장점은 뭐야? [SEP]
    
    *** 정수 인코딩 : [2, 1819, 2472, 2200, 2079, 3676, 1751, 5472, 2073, 1097, 2275, 35, 3]
    
    *** 디코딩 : [CLS] 티베로의 가장 큰 장점은 뭐야? [SEP]
    


## - 연습


```python
encoded = tokenizer.encode('[MASK]') # str() 타입의 문장을 인풋
print('*** 토큰화 결과 :',tokenizer.convert_ids_to_tokens(encoded), end='\n\n')
print('*** <문장으로 보기> :', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded)), end='\n\n')
print('*** 정수 인코딩 :',encoded, end='\n\n') # encoded.ids
print('*** 디코딩 :',tokenizer.decode(encoded), end='\n\n') # encoded.ids
```

    *** 토큰화 결과 : ['[CLS]', '[MASK]', '[SEP]']
    
    *** <문장으로 보기> : [CLS] [MASK] [SEP]
    
    *** 정수 인코딩 : [2, 4, 3]
    
    *** 디코딩 : [CLS] [MASK] [SEP]
    



```python
encoded = tokenizer.encode('서울과 충북 괴산에서 \'국제 청소년포럼\'을 여는 곳은?') # str() 타입의 문장을 인풋
encoded[-3] = 4 # '[MASK]' 토큰
encoded[-4] = 4 # '[MASK]' 토큰

print('*** 토큰화 결과 :',tokenizer.convert_ids_to_tokens(encoded), end='\n\n')
print('*** <문장으로 보기> :', tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded)), end='\n\n')
print('*** 정수 인코딩 :',encoded, end='\n\n') # encoded.ids
print('*** 디코딩 :',tokenizer.decode(encoded), end='\n\n') # encoded.ids
```

    *** 토큰화 결과 : ['[CLS]', '서울', '##과', '충북', '괴산', '##에서', "'", '국제', '청소년', '##포', '##럼', "'", '을', '여', '##는', '[MASK]', '[MASK]', '?', '[SEP]']
    
    *** <문장으로 보기> : [CLS] 서울과 충북 괴산에서'국제 청소년포럼'을 여는 [MASK] [MASK]? [SEP]
    
    *** 정수 인코딩 : [2, 3671, 2145, 7249, 25859, 27135, 11, 3854, 4857, 2208, 2731, 11, 1498, 1428, 2259, 4, 4, 35, 3]
    
    *** 디코딩 : [CLS] 서울과 충북 괴산에서'국제 청소년포럼'을 여는 [MASK] [MASK]? [SEP]
    



```python
# mask_arr = (rand < 0.15) * (inputs.input_ids != 2) * (inputs.input_ids != 3)
# mask_arr
```




    tensor([[False,  True,  True,  ..., False, False, False],
            [False,  True,  True,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            ...,
            [False, False,  True,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False]])




```python
mask_arr
```




    tensor([[False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            ...,
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False]])




```python
inputs.input_ids[0].nonzero()
```




    tensor([[ 0],
            [ 1],
            [ 2],
            [ 3],
            [ 4],
            [ 5],
            [ 6],
            [ 7],
            [ 8],
            [ 9],
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [16],
            [17],
            [18]])




```python
torch.flatten(inputs.input_ids[0].nonzero()).tolist()
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]




```python
corpus_mun[0]
```




    "서울과 충북 괴산에서 '국제 청소년포럼'을 여는 곳은?"




```python
print(len(corpus_mun[0]))
print(len(torch.flatten(mask_arr[0].nonzero()).tolist()))
```

    30
    87



```python
selection = []

for i in range(mask_arr.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:5: UserWarning: This overload of nonzero is deprecated:
    	nonzero()
    Consider using one of the following signatures instead:
    	nonzero(*, bool as_tuple) (Triggered internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:882.)
      """



```python
selection[:10]
```




    [[], [], [], [], [], [], [], [], [], []]




```python
inputs.input_ids[0]
```




    tensor([    2,  3671,  2145,  7249, 25859, 27135,    11,  3854,  4857,  2208,
             2731,    11,  1498,  1428,  2259,   601,  2073,    35,     3,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0])




```python
tokenizer.decode(inputs.input_ids[0])
```




    "[CLS] 서울과 충북 괴산에서'국제 청소년포럼'을 여는 곳은? [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"




```python
# for i in range(mask_arr.shape[0]):
#     inputs.input_ids[i, selection[i]] = 3 # [SEP] 토큰
```


```python
inputs.input_ids[0]
```




    tensor([    2,  3671,  2145,  7249, 25859, 27135,    11,  3854,  4857,  2208,
             2731,    11,  1498,  1428,  2259,   601,  2073,    35,     3,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0])



# - 어텐션 마스크 변형주기


```python
inputs.attention_mask[0]
```




    tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0])




```python
inputs.token_type_ids[0]
```




    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0])




```python
type(inputs.attention_mask[0])
```




    torch.Tensor




```python
inputs.attention_mask.nonzero()[0]
```




    tensor([0, 0])




```python
# (사전 계산) 각 문장들의 토큰 길이 구해놓기
corpus_size = len(corpus_mun)

max_len_list = []

for idx in range(corpus_size):
    encoded = tokenizer.encode(corpus_mun[idx])
    max_len_list.append(len(encoded))
    
max_len_list
```




    [19,
     14,
     8,
     10,
     18,
     22,
     18,
     19,
     16,
     18,
     30,
     25,
     13,
     16,
     13,
     15,
     11,
     18,
     25,
     18,
     18,
     16,
     20,
     16,
     22,
     27,
     21,
     18,
     9,
     43,
     32,
     18,
     57,
     11,
     24,
     28,
     15,
     14,
     27,
     13,
     17,
     15,
     19,
     13,
     19,
     17,
     21,
     31,
     24,
     23,
     21,
     46,
     27,
     13,
     15,
     16,
     38,
     20,
     21,
     15,
     19,
     16,
     12,
     13,
     17,
     14,
     17,
     21,
     15,
     13,
     12,
     13,
     13,
     14,
     14,
     37,
     15,
     12,
     30,
     26,
     29,
     32,
     28,
     19,
     26,
     18,
     25,
     30,
     29,
     19,
     17,
     17,
     22,
     15,
     16,
     12,
     18,
     19,
     16,
     20,
     20,
     14,
     25,
     21,
     15,
     26,
     18,
     26,
     18,
     22,
     63,
     24,
     35,
     40,
     15,
     22,
     47,
     46,
     23,
     20,
     23,
     31,
     27,
     29,
     18,
     12,
     20,
     13,
     12,
     13,
     17,
     21,
     16,
     12,
     19,
     23,
     24,
     16,
     11,
     17,
     23,
     26,
     31,
     17,
     24,
     35,
     27,
     17,
     25,
     16,
     13,
     20,
     20,
     13,
     21,
     17,
     21,
     18,
     13,
     11,
     28,
     22,
     38,
     17,
     40,
     15,
     19,
     12,
     16,
     13,
     18,
     19,
     26,
     21,
     35,
     24,
     21,
     27,
     33,
     25,
     26,
     17,
     19,
     31,
     10,
     17,
     11,
     21,
     9,
     11,
     14,
     15,
     20,
     18,
     20,
     21,
     22,
     11,
     16,
     21,
     11,
     22,
     9,
     14,
     35,
     19,
     11,
     17,
     10,
     26,
     10,
     29,
     37,
     20,
     11,
     24,
     21,
     12,
     19,
     15,
     50,
     14,
     19,
     17,
     17,
     16,
     29,
     25,
     20,
     17,
     21,
     27,
     21,
     22,
     16,
     21,
     11,
     18,
     13,
     16,
     16,
     18,
     18,
     9,
     17,
     21,
     23,
     16,
     18,
     16,
     20,
     28,
     15,
     18,
     18,
     20,
     22,
     10,
     29,
     25,
     32,
     22,
     11,
     32,
     11,
     59,
     23,
     27,
     13,
     20,
     24,
     35,
     15,
     18,
     21,
     12,
     33,
     22,
     20,
     27,
     41,
     18,
     19,
     27,
     50,
     18,
     14,
     14,
     15,
     27,
     24,
     24,
     22,
     12,
     16,
     25,
     27,
     62,
     70,
     12,
     12,
     11,
     26,
     13,
     16,
     13,
     13,
     16,
     17,
     29,
     18,
     24,
     17,
     20,
     19,
     20,
     21,
     34,
     16,
     10,
     20,
     17,
     13,
     13,
     23,
     19,
     16,
     18,
     16,
     20,
     10,
     17,
     20,
     17,
     25,
     14,
     26,
     19,
     23,
     20,
     25,
     30,
     11,
     28,
     25,
     28,
     24,
     24,
     16,
     16,
     22,
     11,
     15,
     25,
     17,
     14,
     18,
     21,
     41,
     64,
     19,
     11,
     25,
     15,
     22,
     19,
     37,
     32,
     24,
     19,
     14,
     17,
     17,
     22,
     16,
     9,
     15,
     19,
     17,
     20,
     18,
     15,
     13,
     22,
     27,
     14,
     14,
     30,
     19,
     20,
     17,
     15,
     14,
     14,
     16,
     20,
     17,
     18,
     26,
     19,
     23,
     23,
     16,
     20,
     9,
     16,
     16,
     16,
     10,
     13,
     19,
     20,
     16,
     35,
     12,
     23,
     22,
     37,
     29,
     38,
     34,
     19,
     74,
     57,
     17,
     23,
     20,
     16,
     24,
     27,
     24,
     26,
     17,
     10,
     24,
     11,
     20,
     23,
     16,
     10,
     14,
     17,
     13,
     19,
     12,
     19,
     19,
     19,
     21,
     21,
     31,
     30,
     25,
     30,
     31,
     31,
     19,
     9,
     45,
     18,
     25,
     23,
     22,
     17,
     18,
     12,
     19,
     18,
     23,
     22,
     25,
     12,
     33,
     12,
     13,
     12,
     11,
     9,
     15,
     10,
     16,
     10,
     21,
     18,
     19,
     21,
     15,
     38,
     23,
     33,
     19,
     26,
     20,
     9,
     21,
     38,
     43,
     17,
     34,
     26,
     18,
     19,
     12,
     20,
     22,
     10,
     13,
     11,
     13,
     16,
     19,
     28,
     32,
     22,
     17,
     18,
     24,
     17,
     15,
     39,
     14,
     13,
     20,
     11,
     17,
     14,
     13,
     20,
     18,
     12,
     15,
     16,
     18,
     19,
     17,
     24,
     33,
     27,
     28,
     11,
     23,
     15,
     15,
     14,
     10,
     14,
     13,
     24,
     18,
     12,
     18,
     21,
     24,
     20,
     22,
     18,
     19,
     19,
     26,
     21,
     18,
     19,
     21,
     34,
     46,
     30,
     33,
     22,
     19,
     23,
     21,
     19,
     14,
     14,
     13,
     11,
     15,
     16,
     14,
     20,
     27,
     34,
     20,
     19,
     16,
     29,
     19,
     13,
     32,
     32,
     17,
     23,
     26,
     22,
     20,
     16,
     20,
     21,
     22,
     18,
     33,
     28,
     24,
     35,
     19,
     38,
     34,
     11,
     26,
     16,
     17,
     21,
     31,
     11,
     26,
     21,
     15,
     17,
     15,
     15,
     27,
     20,
     14,
     22,
     21,
     21,
     16,
     16,
     25,
     26,
     18,
     20,
     15,
     15,
     17,
     15,
     16,
     26,
     11,
     25,
     25,
     26,
     17,
     30,
     22,
     10,
     11,
     13,
     23,
     10,
     10,
     45,
     21,
     21,
     22,
     15,
     18,
     18,
     15,
     15,
     20,
     17,
     18,
     30,
     20,
     22,
     28,
     10,
     21,
     15,
     15,
     12,
     18,
     14,
     54,
     50,
     65,
     27,
     30,
     29,
     13,
     19,
     16,
     18,
     19,
     16,
     30,
     24,
     27,
     24,
     15,
     17,
     23,
     21,
     25,
     13,
     24,
     17,
     21,
     16,
     18,
     21,
     17,
     23,
     22,
     20,
     19,
     14,
     22,
     16,
     20,
     44,
     15,
     22,
     15,
     23,
     14,
     14,
     19,
     22,
     15,
     17,
     13,
     17,
     10,
     15,
     23,
     20,
     15,
     18,
     18,
     32,
     17,
     25,
     27,
     47,
     17,
     30,
     17,
     28,
     22,
     38,
     32,
     29,
     35,
     36,
     29,
     29,
     24,
     20,
     20,
     6,
     15,
     14,
     14,
     16,
     23,
     27,
     13,
     16,
     25,
     15,
     20,
     21,
     22,
     26,
     19,
     21,
     17,
     25,
     23,
     24,
     29,
     25,
     26,
     28,
     30,
     26,
     27,
     36,
     21,
     35,
     34,
     41,
     22,
     22,
     13,
     17,
     18,
     17,
     15,
     17,
     15,
     14,
     10,
     16,
     11,
     17,
     11,
     17,
     13,
     16,
     18,
     30,
     14,
     20,
     30,
     22,
     17,
     18,
     9,
     17,
     21,
     28,
     13,
     21,
     20,
     24,
     17,
     31,
     22,
     21,
     21,
     19,
     18,
     21,
     19,
     43,
     44,
     20,
     47,
     24,
     30,
     30,
     20,
     21,
     16,
     12,
     26,
     25,
     14,
     15,
     16,
     11,
     19,
     16,
     15,
     19,
     22,
     16,
     46,
     15,
     19,
     21,
     16,
     13,
     17,
     23,
     23,
     15,
     37,
     31,
     10,
     23,
     17,
     19,
     13,
     15,
     16,
     24,
     15,
     15,
     16,
     16,
     27,
     21,
     27,
     20,
     51,
     14,
     29,
     13,
     17,
     22,
     12,
     20,
     18,
     12,
     28,
     41,
     13,
     15,
     21,
     16,
     18,
     23,
     28,
     17,
     25,
     13,
     21,
     17,
     21,
     19,
     22,
     26,
     16,
     23,
     11,
     25,
     23,
     31,
     12,
     28,
     17,
     16,
     28,
     16,
     21,
     22,
     19,
     33,
     18,
     23,
     21,
     13,
     14,
     15,
     32,
     27,
     28,
     17,
     24,
     16,
     52,
     60,
     41,
     14,
     42,
     13,
     11,
     28,
     13,
     35,
     37,
     19,
     39,
     40,
     27,
     23,
     18,
     19,
     26,
     31,
     19,
     18,
     21,
     30,
     47,
     13,
     14,
     23,
     17,
     17,
     12,
     9,
     7,
     13,
     14,
     18,
     20,
     23,
     19,
     43,
     24,
     17,
     20,
     25,
     22,
     25,
     32,
     33,
     18,
     19,
     30,
     21,
     32,
     19,
     20,
     29,
     13,
     18,
     15,
     13,
     19,
     17,
     14,
     11,
     9,
     26,
     ...]




```python
# selection = []

# for i in range(mask_arr.shape[0]):
#     selection.append(
#         torch.flatten(mask_arr[i].nonzero()).tolist()
#     )
```


```python
mask_arr
```




    tensor([[False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            ...,
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False],
            [False, False, False,  ..., False, False, False]])




```python
mask_arr[0][19] = True
```


```python
mask_arr[0] # 변화 적용됨을 확인함 !
```




    tensor([False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,  True,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False])




```python
mask_arr[0][19-2 : 19+4] = True
```


```python
mask_arr[0]
```




    tensor([False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False,  True,  True,  True,
             True,  True,  True, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False])




```python
mask_arr.shape
```




    torch.Size([196327, 128])




```python
inputs
```




    {'input_ids': tensor([[   2, 3671, 2145,  ...,    0,    0,    0],
            [   2,   11, 3854,  ...,    0,    0,    0],
            [   2, 3686, 6431,  ...,    0,    0,    0],
            ...,
            [   2, 5217, 2440,  ...,    0,    0,    0],
            [   2, 7275, 6551,  ...,    0,    0,    0],
            [   2, 5217, 2440,  ...,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            ...,
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0]]), 'labels': tensor([[   2, 3671, 2145,  ...,    0,    0,    0],
            [   2,   11, 3854,  ...,    0,    0,    0],
            [   2, 3686, 6431,  ...,    0,    0,    0],
            ...,
            [   2, 5217, 2440,  ...,    0,    0,    0],
            [   2, 7275, 6551,  ...,    0,    0,    0],
            [   2, 5217, 2440,  ...,    0,    0,    0]])}




```python
print(inputs.input_ids[0, 0:19-4])
print(inputs.attention_mask[0, 0:19-4])
print()
print(inputs.input_ids[0, 19-4:19+4])
print(inputs.attention_mask[0, 19-4:19+4])
```

    tensor([    2,  3671,  2145,  7249, 25859, 27135,    11,  3854,  4857,  2208,
             2731,    11,  1498,  1428,  2259])
    tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    tensor([ 601, 2073,   35,    3,    0,    0,    0,    0])
    tensor([1, 1, 1, 1, 0, 0, 0, 0])



```python
for i in range(corpus_size):
    leng = max_len_list[i]
    inputs.attention_mask[i, 0:leng-4] = 0 
    inputs.attention_mask[i, leng-4:leng+4] = 1 # 1 로 바꿈 (masking해서 가르칠 대상)
```


```python
print(inputs.input_ids[0, 0:19-4])
print(inputs.attention_mask[0, 0:19-4])
print()
print(inputs.input_ids[0, 19-4:19+4])
print(inputs.attention_mask[0, 19-4:19+4])
```

    tensor([    2,  3671,  2145,  7249, 25859, 27135,    11,  3854,  4857,  2208,
             2731,    11,  1498,  1428,  2259])
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    tensor([ 601, 2073,   35,    3,    0,    0,    0,    0])
    tensor([1, 1, 1, 1, 1, 1, 1, 1])



```python
inputs.attention_mask[0]
```




    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0])




```python

```


```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
```


```python
dataset = Dataset(inputs)
```


```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True) ###
```


```python
##device = torch.device('cuda:0')
torch.cuda.is_available()
```




    True




```python
device = ('cuda:0')

model.to(device)
```




    BertForMaskedLM(
      (bert): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(32000, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (2): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (3): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (4): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (5): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (6): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (7): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (8): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (9): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (10): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (11): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (cls): BertOnlyMLMHead(
        (predictions): BertLMPredictionHead(
          (transform): BertPredictionHeadTransform(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (transform_act_fn): GELUActivation()
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          )
          (decoder): Linear(in_features=768, out_features=32000, bias=True)
        )
      )
    )




```python
model.train()
```




    BertForMaskedLM(
      (bert): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(32000, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (2): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (3): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (4): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (5): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (6): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (7): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (8): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (9): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (10): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (11): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (cls): BertOnlyMLMHead(
        (predictions): BertLMPredictionHead(
          (transform): BertPredictionHeadTransform(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (transform_act_fn): GELUActivation()
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          )
          (decoder): Linear(in_features=768, out_features=32000, bias=True)
        )
      )
    )




```python
optim = AdamW(model.parameters(), lr=1e-5)

from tqdm import tqdm

epochs = 5 ####
for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        
    
    
    
    #####model.save_pretrained('model/MLM_train_{0}_epoch.pt'.format(epoch))
    print(epoch)
    print(loss)
    print('* * * * *')
    ####PATH = 'model/MLM_train_mun_{0}_epoch'.format(epoch)
    PATH = 'model/padded_MLM_train_mun_{0}_epoch'.format(epoch)
    torch.save(model, PATH)
```

      0%|          | 0/6136 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:5: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      """
    Epoch 0: 100%|██████████| 6136/6136 [32:28<00:00,  3.15it/s, loss=1.46e-5] 


    0
    tensor(1.4593e-05, device='cuda:0', grad_fn=<NllLossBackward>)
    * * * * *


    Epoch 1: 100%|██████████| 6136/6136 [32:33<00:00,  3.14it/s, loss=4.01e-6] 


    1
    tensor(4.0064e-06, device='cuda:0', grad_fn=<NllLossBackward>)
    * * * * *


    Epoch 2: 100%|██████████| 6136/6136 [32:33<00:00,  3.14it/s, loss=4.14e-6]


    2
    tensor(4.1411e-06, device='cuda:0', grad_fn=<NllLossBackward>)
    * * * * *


    Epoch 3: 100%|██████████| 6136/6136 [32:33<00:00,  3.14it/s, loss=1.47e-6]


    3
    tensor(1.4723e-06, device='cuda:0', grad_fn=<NllLossBackward>)
    * * * * *


    Epoch 4: 100%|██████████| 6136/6136 [32:33<00:00,  3.14it/s, loss=1.07e-6]


    4
    tensor(1.0654e-06, device='cuda:0', grad_fn=<NllLossBackward>)
    * * * * *



```python
# #### Fill-Mask 파이프라인
# '''
# Masked Language Modeling은 문장의 일부 단어를 마스킹하고 그 마스크를 대체해야 하는 단어를 예측하는 작업입니다. 이 모델은 모델이 훈련된 언어에 대한 통계적 이해를 얻고자 할 때 유용합니다.
# '''

# from transformers import pipeline

# classifier = pipeline("fill-mask")
# classifier("Paris is the <mask> of France.")

# # [{'score': 0.7, 'sequence': 'Paris is the capital of France.'},
# # {'score': 0.2, 'sequence': 'Paris is the birthplace of France.'},
# # {'score': 0.1, 'sequence': 'Paris is the heart of France.'}]
```

# - MASKING된 문장 예측하기


```python
# 1. 모델 기 학습한 거 load 하기
# weights_path = "model/MLM_train_ban_4_epoch" #"model/MLM_train_2_epoch"
# model_result = torch.load(weights_path)
# ###print(model_result)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-34-b64c4b05f223> in <module>
          1 # 1. 모델 기 학습한 거 load 하기
          2 weights_path = "model/MLM_train_ban_4_epoch" #"model/MLM_train_2_epoch"
    ----> 3 model_result = torch.load(weights_path)
          4 ###print(model_result)


    ~/.local/lib/python3.6/site-packages/torch/serialization.py in load(f, map_location, pickle_module, **pickle_load_args)
        592                     opened_file.seek(orig_position)
        593                     return torch.jit.load(opened_file)
    --> 594                 return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
        595         return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
        596 


    ~/.local/lib/python3.6/site-packages/torch/serialization.py in _load(zip_file, map_location, pickle_module, pickle_file, **pickle_load_args)
        851     unpickler = pickle_module.Unpickler(data_file, **pickle_load_args)
        852     unpickler.persistent_load = persistent_load
    --> 853     result = unpickler.load()
        854 
        855     torch._utils._validate_loaded_sparse_tensors()


    ModuleNotFoundError: No module named 'torch._C._nn'; 'torch._C' is not a package



```python

```
