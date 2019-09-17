import coremltools
import coremltools.models.datatypes as datatypes
import numpy as np
import torch
from coremltools.models import neural_network as neural_network
from coremltools.models.utils import save_spec
# get weights
from pytorch_transformers.modeling_distilbert import (DistilBertConfig,
                                                      DistilBertModel,
                                                      TransformerBlock)

model = DistilBertModel.from_pretrained('distilbert-base-uncased-distilled-squad')
config: DistilBertConfig = model.config

sequence_length = config.max_position_embeddings  # 512
steps = config.n_layers  # 6


# build model
input_features = [
	('input_ids', datatypes.Array(sequence_length)),
]
output_features = [
	('output_logits', None)
]

builder = neural_network.NeuralNetworkBuilder(
	input_features,
	output_features,
	mode=None,
	disable_rank5_shape_mapping=True,
)
builder.add_expand_dims(
	name='input_ids_expanded_to_rank5',
	input_name='input_ids',
	output_name='input_ids_expanded_to_rank5',
	axes=(1, 2, 3, 4)
)
builder.add_embedding(
	name='token_embeddings',
	input_name='input_ids_expanded_to_rank5',
	output_name='token_embeddings',
	W=model.embeddings.word_embeddings.weight.data.numpy().transpose(), # shape (768, 30522)
	b=None,
	input_dim=config.vocab_size,
	output_channels=768,
	has_bias=False,
)
builder.add_mvn(
	name='embeddings_ln',
	input_name=f"token_embeddings",
	output_name=f"embeddings_ln",
	across_channels=True,
	normalize_variance=True,
	epsilon=model.embeddings.LayerNorm.eps
)
builder.add_scale(
	name=f"embeddings_ln_scaled",
	input_name=f"embeddings_ln",
	output_name=f'{0}_previous_block',
	# output_name=f'output_logits',
	W=model.embeddings.LayerNorm.weight.data.numpy().reshape((1, 1, 768, 1, 1)),
	b=model.embeddings.LayerNorm.bias.data.numpy().reshape((1, 1, 768, 1, 1)),
	has_bias=True,
	shape_scale=[768],
	shape_bias=[768]
)

for i in range(steps):
	print(i)
	layer: TransformerBlock = model.transformer.layer[i]
	
	# MultiHeadSelfAttention
	## wip
	# sa_layer_norm
	builder.add_mvn(
		name=f"{i}_block_ln_2",
		input_name=f"{i}_block_xa_sum",
		output_name=f"{i}_block_ln_2",
		across_channels=True,
		normalize_variance=True,
		epsilon=layer.sa_layer_norm.eps
	)
	builder.add_scale(
		name=f"{i}_block_ln_2_scaled",
		input_name=f"{i}_block_ln_2",
		output_name=f"{i}_block_ln_2_scaled",
		W=layer.sa_layer_norm.weight.data.numpy().reshape((1, 1, 768, 1, 1)),
		b=layer.sa_layer_norm.bias.data.numpy().reshape((1, 1, 768, 1, 1)),
		has_bias=True,
		shape_scale=[768],
		shape_bias=[768]
	)
	
	# Feed Forward Network
	builder.add_inner_product(
		name=f"{i}_block_mlp_conv_fc",
		input_name=f"{i}_block_ln_2_scaled",
		output_name=f"{i}_block_mlp_conv_fc",
		input_channels=768,
		output_channels=3072,
		W=layer.ffn.lin1.weight.data.numpy().transpose().reshape((1, 768, 3072, 1, 1)),
		b=layer.ffn.lin1.bias.data.numpy().reshape((1, 1, 3072, 1, 1)),
		has_bias=True
	)
	builder.add_gelu(
		name=f"{i}_block_mlp_gelu",
		input_name=f"{i}_block_mlp_conv_fc",
		output_name=f"{i}_block_mlp_gelu",
		mode='TANH_APPROXIMATION'
	)
	builder.add_inner_product(
		name=f"{i}_block_mlp_conv_proj",
		input_name=f"{i}_block_mlp_gelu",
		output_name=f"{i}_block_mlp_conv_proj",
		input_channels=3072,
		output_channels=768,
		W=layer.ffn.lin2.weight.data.numpy().transpose().reshape((1, 3072, 768, 1, 1)),
		b=layer.ffn.lin2.bias.data.numpy().reshape((1, 1, 768, 1, 1)),
		has_bias=True
	)
	
	# output_layer_norm
	# Input: (1, seq, 768, 1, 1), Output:
	builder.add_mvn(
		name=f"{i}_output_ln",
		input_name=f"{i}_block_mlp_conv_proj",
		output_name=f"{i}_output_ln",
		across_channels=True,
		normalize_variance=True,
		epsilon=layer.output_layer_norm.eps
	)
	builder.add_scale(
		name=f"{i}_output_ln_scaled",
		input_name=f"{i}_output_ln",
		output_name=f"{i}_output_ln_scaled",
		W=layer.output_layer_norm.weight.data.numpy().reshape((1, 1, 768, 1, 1)),
		b=layer.output_layer_norm.bias.data.numpy().reshape((1, 1, 768, 1, 1)),
		has_bias=True,
		shape_scale=[768],
		shape_bias=[768]
	)



# compile spec to model
# mlmodel = coremltools.models.MLModel(builder.spec)
# input_data = {
# 	'input_ids': np.array([ 7592, 1010, 2026, 3899, 2003, 10140 ])
# }
# predictions = mlmodel.predict(input_data)["output_logits"]
# print(predictions)


save_spec(builder.spec, f'../Resources/distilbert-{sequence_length}-{steps}.mlmodel')
