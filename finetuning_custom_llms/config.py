# config.py

import torch

# Optimization options
use_quantization = True  # Enable quantization to reduce memory usage
quantization_method = 'bitsandbytes'  # Options: 'bitsandbytes', 'gptq'

# Model options
model_name = 'tiiuae/falcon-40b-instruct'

# Device options
device = 'cuda'  # 'cuda' or 'cpu'

# Torch data type
torch_dtype = torch.float16  # Use float16 for reduced memory footprint

# Cache directory for HuggingFace models
cache_dir = '/scratch/gilbreth/bhattar1/.cache/huggingface/transformers/codeql_falcon/'

# Other options (commented out for future tweaking)
# max_length = 2048
# temperature = 0.2
# top_k = 10
