# config.py

# Optimization options
use_quantization = False  # Set to True to enable quantization
quantization_method = 'bitsandbytes'  # Options: 'bitsandbytes', 'gptq'

# Model options
model_name = 'tiiuae/falcon-40b-instruct'

# Device options
device = 'cuda'  # 'cuda' or 'cpu'

# Torch data type
torch_dtype = torch.bfloat16

# Other options (commented out for future tweaking)
# max_length = 2048
# temperature = 0.2
# top_k = 10
