"""
Credits: 

Bhushan Sonawane https://github.com/bhushan23 Apple, Inc.

https://github.com/onnx/onnx-coreml/issues/478

"""
from pytorch_transformers.modeling_distilbert import DistilBertForQuestionAnswering
from onnx_coreml import convert
import torch
import numpy as np

model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad", torchscript=True
)
torch.save(model, './distilbert.pt')
model.eval()

torch.onnx.export(
    model,
    torch.ones(1, 128, dtype=torch.long),
    "distilbert-squad-128.onnx",
    verbose=True,
    input_names=["input_ids"],
    output_names=["start_scores", "end_scores"],
)


def _convert_softmax(builder, node, graph, err):
    """
    convert to CoreML SoftMax ND Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#3547
    """
    axis = node.attrs.get("axis", 1)
    builder.add_softmax_nd(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
        + ("_softmax" if node.op_type == "LogSoftmax" else ""),
        axis=axis,
    )
    if node.op_type == "LogSoftmax":
        builder.add_unary(
            name=node.name + "_log",
            input_name=node.outputs[0] + "_softmax",
            output_name=node.outputs[0],
            mode="log",
        )


mlmodel = convert(
    model="./distilbert-squad-128.onnx",
    target_ios="13",
    custom_conversion_functions={"Softmax": _convert_softmax},
)
mlmodel.save("./distilbert-squad-128.mlmodel")
