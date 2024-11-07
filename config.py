# config.py

import torch

# Optimization options
use_quantization = False  # Set to True to enable quantization
quantization_method = 'bitsandbytes'  # Options: 'bitsandbytes', 'gptq'

# Model options
model_name = 'tiiuae/falcon-40b-instruct'

# Device options
device = 'cuda'  # 'cuda' or 'cpu'

# Torch data type
torch_dtype = torch.bfloat16  # or torch.float16 if bfloat16 is not supported

# Cache directory for HuggingFace models
cache_dir = '/scratch/gilbreth/bhattar1/.cache/huggingface/transformers/codeql_llama/'

# Other options (commented out for future tweaking)
# max_length = 2048
# temperature = 0.2
# top_k = 10
