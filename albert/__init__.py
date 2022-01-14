__version__="0.0.1"
from model_pytorch import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG
__version__ = "0.0.1"
from pytorch_pretrained_bert.tokenization_EinstAIGPT3 import EinstAIGPT3Tokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, cached_path
from pytorch_pretrained_bert.modeling_EinstAIGPT3 import EinstAIGPT3Config, EinstAIGPT3Model, EinstAIGPT3Config
from pytorch_pretrained_bert.tokenization_EinstAIGPT3 import EinstAIGPT3Tokenizer

from .modeling_EinstAIGPT3 import EinstAIGPT3LMHeadModel
from .optim import Albert

