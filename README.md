# Project
## Description
This is a mini version of GPT model, building generative transformer model based on tokenizer and neural network

The project is inspired by [nonoGPT](https://github.com/karpathy/nanoGPT/tree/master) by @karpathy

## Structure

```
nano-model
├── input
|  ├── romance_of_three_kindoms.txt
|  └── tale_of_twin_cities.txt
├── model.pth
├── model_dropout.pth
├── model_tokenizer.pth
├── README.md
├── requirements.txt
├── train.py
├── train_bleu.py
├── train_with_dropout.py
├── train_with_tokenizer.py
├── utility
|  ├── default_tokenizer.py
|  ├── tiktoken_tokenizer.py
|  ├── tokenizer.py
|  ├── __init__.py
|  └── __pycache__
├── v2.py
```

## Commands
To start the virtual environment, run `venv\Scripts\activate` in Command Prompt in windows

To generate dependencies, run `pip freeze > requirements.txt`

To download dependencies, run `pip install -r requirements.txt`

## Reference

1. Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N; Kaiser, Łukasz; Polosukhin, Illia (2017). Attention is All you Need. *Advances in Neural Information Processing Systems*.
2. Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun (2014). Deep Residual Learning for Image Recognition. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.