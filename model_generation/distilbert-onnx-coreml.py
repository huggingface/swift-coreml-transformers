import timeit

import torch

from transformers.modeling_bert import BertForMaskedLM, BertModel
from transformers.modeling_distilbert import (
    DistilBertModel,
    DistilBertForQuestionAnswering,
)
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer


tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)
input_ids = torch.tensor(
    [
        tokenizer.encode(
            "Here is some text to encode, Here is some text to encode, Here is some text to encode",
            add_special_tokens=True,
        )
    ],
    dtype=torch.long,
)
print(input_ids.shape)
model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)
print(model(input_ids)[0].shape)

print(timeit.repeat(lambda: model(input_ids), number=1))
print()


model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad", torchscript=True
)
model.eval()
traced_model = torch.jit.trace(model, (input_ids,))
print(timeit.repeat(lambda: traced_model(input_ids), number=1))

torch.onnx.export(
    model,
    torch.ones(1, 128, dtype=torch.long),
    "distilbert-squad-128.onnx",
    verbose=True,
    input_names=["input_ids"],
    output_names=["start_scores", "end_scores"],
)


print()
from onnx_coreml import convert

mlmodel = convert(model="distilbert-squad-128.onnx", target_ios="13")  # to use CoreML 3
mlmodel.save("distilbert-squad-128.mlmodel")


# torch.Size([1, 25, 30522])
# [0.11437926300000001, 0.10830704900000043, 0.12072527099999952, 0.11071877800000074, 0.1034001730000007]

# [0.6870135510000015, 0.09324176099999981, 0.09249538999999984, 0.09238876399999896, 0.09471561299999998]

