
# Swift Core ML implementations of Transformers: GPT-2, BERT, more coming soon!

This repository contains:
- For **BERT**:
	- a pretrained [Google BERT model](https://github.com/google-research/bert) fine-tuned for Question answering on the SQuAD dataset.
	- Swift implementations of the [BERT tokenizer](https://github.com/huggingface/swift-coreml-transformers/blob/master/Sources/BertTokenizer.swift) (`BasicTokenizer` and `WordpieceTokenizer`) and SQuAD dataset parsing utilities.
	- A demo question answering app.
- For **GPT-2**:
	- a [conversion script](https://github.com/huggingface/swift-coreml-transformers/blob/master/model_generation/gpt2.py) from Pytorch trained GPT-2 models (see our [`pytorch-transformers`](https://github.com/huggingface/pytorch-transformers) repo) to CoreML models.
	- The [GPT-2 generation model](https://github.com/huggingface/swift-coreml-transformers/blob/master/Sources/GPT2.swift) itself, including decoding strategies (greedy and TopK are currently implemented) and GPT-2 Byte-pair encoder and decoder.
	- A neat demo app showcasing on-device text generation.


# 🦄 GPT-2

Unleash the full power of text generation with GPT-2 on device!!

![demo](https://raw.githubusercontent.com/huggingface/swift-coreml-transformers/master/media/coreml-gpt2.gif)

# 🐸 BERT

The pretrained Core ML model was packaged by Apple and is linked from the [main ML models page](https://developer.apple.com/machine-learning/models/#text). It was demoed at WWDC 2019 as part of the Core ML 3 launch.

![core ml 3](https://raw.githubusercontent.com/huggingface/swift-coreml-transformers/master/media/coreml3-models-tweaked.png)

## 🦄 Demo Time 🔥

![demo](https://raw.githubusercontent.com/huggingface/swift-coreml-transformers/master/media/coreml-squad-small.gif)

Apple demo at WWDC 2019

![wwdc demo](https://raw.githubusercontent.com/huggingface/swift-coreml-transformers/master/media/wwdc704.gif)

full video [here](https://developer.apple.com/videos/play/wwdc2019/704)

## BERT Architecture (wwdc slide)

![bert](https://raw.githubusercontent.com/huggingface/swift-coreml-transformers/master/media/bert-architecture.png)

## Notes

We use `git-lfs` to store large model files and it is required to obtain some of the files the app needs to run.
See how to install `git-lfs`on the [installation page](https://git-lfs.github.com/)

