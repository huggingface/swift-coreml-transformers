
# Swift Core ML implementation of BERT

This repository contains:
- a pretrained [Google BERT model](https://github.com/google-research/bert) fine-tuned for Question answering on the SQuAD dataset.
- Swift implementations of the [BERT tokenizer](https://github.com/huggingface/swift-coreml-transformers/blob/master/Sources/BertTokenizer.swift) (`BasicTokenizer` and `WordpieceTokenizer`) and SQuAD dataset parsing utilities.
- A demo question answering app.

The pretrained Core ML model was packaged by Apple and is linked from the [main ML models page](https://developer.apple.com/machine-learning/models/#text). It was demoed at WWDC 2019 as part of the Core ML 3 launch.

## 🦄 Demo Time 🔥


### Notes

We use `git-lfs` to store large model files and it is required to obtain some of the files the app needs to run.
See how to install `git-lfs`on the [installation page](https://git-lfs.github.com/)

