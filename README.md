# Project
## Description
This is a mini version of GPT model, building generative transformer model based on tokenizer and neural network

The project is inspired by [nonoGPT](https://github.com/karpathy/nanoGPT/tree/master) by @karpathy

The latest version is train_sschedule.py

## Structure

```
nano-model
├── demo
├── gpt2.py
├── input
|  ├── david_copperfield.txt
|  ├── great_expectations.txt
|  ├── les_miserables.txt
|  ├── oliver_twist.txt
|  ├── romance_of_three_kindoms.txt
|  ├── tale_of_two_cities.txt
|  ├── the_count_of_monte_cristo.txt
|  ├── the_three_musketeers.txt
|  └── war_and_peace.txt
├── model
|  ├── best_model.pth
|  ├── best_model_val_loss.txt
|  ├── model.pth
|  ├── model_dropout.pth
|  └── model_tokenizer.pth
├── README.md
├── requirements.txt
├── train.py
├── train_bleu.py
├── train_multiple_files.py
├── train_schedule.py
├── train_with_dropout.py
├── train_with_tokenizer.py
├── utility
|  ├── default_tokenizer.py
|  ├── gpt2_tokenizer.py
|  ├── tiktoken_tokenizer.py
|  ├── tokenizer.py
|  ├── __init__.py
```

## Commands
To start the virtual environment, run `venv\Scripts\activate` in Command Prompt in windows

To generate dependencies, run `pip freeze > requirements.txt`

To download dependencies, run `pip install -r requirements.txt`

## Reference

1. Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N; Kaiser, Łukasz; Polosukhin, Illia (2017). Attention is All you Need. *Advances in Neural Information Processing Systems*.
2. Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun (2014). Deep Residual Learning for Image Recognition. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.