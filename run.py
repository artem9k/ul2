from t5.test import Test
from t5.t5 import T5ForConditionalGeneration
from transformers import PretrainedConfig

# create model from pretrained
model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-small')
print('model loaded')
