from t5.test import Test
from t5.t5 import T5ForConditionalGeneration
print('hello, sanity check!')

config = None
T5ForConditionalGeneration(config)

print('model loaded')