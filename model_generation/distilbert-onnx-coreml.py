"""
PyTorch -> onnx -> coreml conversion

Credits: 

Bhushan Sonawane https://github.com/bhushan23 Apple, Inc.

https://github.com/onnx/onnx-coreml/issues/478

"""
import os
import timeit
import numpy as np
import torch
import coremltools
from onnx_coreml import convert
from transformers.modeling_distilbert import DistilBertForQuestionAnswering
from transformers.tokenization_distilbert import DistilBertTokenizer
from utils import _compute_SNR

SEQUENCE_LENGTH = 384
MODEL_NAME = "distilbert-base-uncased-distilled-squad"


model = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME, torchscript=True)
# torch.save(model, "./distilbert.pt")
model.eval()

torch.onnx.export(
    model,
    torch.ones(1, SEQUENCE_LENGTH, dtype=torch.long),
    f"./distilbert-squad-{SEQUENCE_LENGTH}.onnx",
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
    model=f"./distilbert-squad-{SEQUENCE_LENGTH}.onnx",
    target_ios="13",
    custom_conversion_functions={"Softmax": _convert_softmax},
)
mlmodel.save(f"../Resources/distilbert-squad-{SEQUENCE_LENGTH}.mlmodel")
os.remove(f"./distilbert-squad-{SEQUENCE_LENGTH}.onnx")

# fp16
try:
    model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(
        mlmodel.get_spec()
    )
    coremltools.utils.save_spec(
        model_fp16_spec, f"../Resources/distilbert-squad-{SEQUENCE_LENGTH}_FP16.mlmodel"
    )
except Exception as e:
    print(e)

##### Now check the outputs.
print("––––––\n")

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)


def generate_input_ids() -> np.array:
    """
    Returns:
        np.array of shape (1, seq_len)
    """
    x = tokenizer.encode(
        "Here is some text to encode, Here is some text to encode, Here is some text to encode",
        add_special_tokens=True,
    )
    x += (SEQUENCE_LENGTH - len(x)) * [tokenizer.pad_token_id]
    return np.array([x], dtype=np.long)


input_ids = generate_input_ids()
outputs_pt = model(torch.tensor(input_ids))
outputs_pt = (outputs_pt[0].detach().numpy(), outputs_pt[1].detach().numpy())

pred_coreml = mlmodel.predict(
    {"input_ids": input_ids.astype(np.float32)}, useCPUOnly=True
)

snr = _compute_SNR(pred_coreml["start_scores"], outputs_pt[0])
print(f"Start Scores: SNR, PSNR {snr}")
snr = _compute_SNR(pred_coreml["end_scores"], outputs_pt[1])
print(f"End Scores: SNR, PSNR {snr}")


##### Perf benchmark.
print("––––––\n")


def timeit_and_report_mean(f) -> str:
    samples = timeit.repeat(f, number=1)
    print(samples)
    report = f"{np.mean(samples)} s ± {np.std(samples)} s per execution"
    return report


print("PyTorch", timeit_and_report_mean(lambda: model(torch.tensor(input_ids))))
# Here we could also benchmark torchscript-traced version of `model`
print(
    "CoreML",
    timeit_and_report_mean(
        lambda: mlmodel.predict(
            {"input_ids": input_ids.astype(np.float32)}, useCPUOnly=True
        )
    ),
)

