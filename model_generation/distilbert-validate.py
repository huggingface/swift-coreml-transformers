"""
Credits: 

Bhushan Sonawane https://github.com/bhushan23 Apple, Inc.

https://github.com/onnx/onnx-coreml/issues/478

"""
import onnx
import onnxruntime as rt
import coremltools
import torch
import numpy as np
from utils import _compute_SNR


spec = coremltools.utils.load_spec("../Resources/distilbert-squad-128.mlmodel")
mlmodel = coremltools.models.MLModel(spec, useCPUOnly=True)

input = np.random.randint(0, high=1000, size=(1, 128))
input_dict = {"input_ids": input.astype(np.float32)}

pred_coreml = mlmodel.predict(input_dict, useCPUOnly=True)

model = torch.load("./distilbert.pt")
pred_pt = model(torch.from_numpy(input).type(torch.LongTensor))

pt_out = {}
pt_out["start_scores"] = pred_pt[0].detach().numpy()
pt_out["end_scores"] = pred_pt[1].detach().numpy()

snr, psnr = _compute_SNR(pred_coreml["start_scores"], pt_out["start_scores"])
print("Start Scores: SNR {}, PSNR {}".format(snr, psnr))
snr, psnr = _compute_SNR(pred_coreml["end_scores"], pt_out["end_scores"])
print("End Scores: SNR {}, PSNR {}".format(snr, psnr))
