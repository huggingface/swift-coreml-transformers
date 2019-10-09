"""
Recreate the Core ML model from scratch using
coremltools' neural_network.NeuralNetworkBuilder
"""
import coremltools
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
from coremltools.models.utils import save_spec
import numpy as np

# get weights
from transformers import GPT2LMHeadModel
model_name = "gpt2"
lm_head_model = GPT2LMHeadModel.from_pretrained(model_name)
model = lm_head_model.transformer

wte = model.wte.weight.data.numpy().transpose() # shape (768, 50257) /!\ i hate this
wpe = model.wpe.weight.data.numpy().transpose() # shape (768, 1024)

sequence_length = 64
steps = 12

# build model
input_features = [
	('input_ids', datatypes.Array(sequence_length)),
	('position_ids', datatypes.Array(sequence_length)),
]
output_features = [('output_logits', None)]

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
builder.add_expand_dims(
	name='position_ids_expanded_to_rank5',
	input_name='position_ids',
	output_name='position_ids_expanded_to_rank5',
	axes=(1, 2, 3, 4)
)
builder.add_embedding(
	name='token_embeddings',
	input_name='input_ids_expanded_to_rank5',
	output_name='token_embeddings',
	W=wte,
	b=None,
	input_dim=50257,
	output_channels=768,
	has_bias=False,
)
builder.add_embedding(
	name='positional_embeddings',
	input_name='position_ids_expanded_to_rank5',
	output_name='positional_embeddings',
	W=wpe,
	b=None,
	input_dim=1024,
	output_channels=768,
	has_bias=False,
)

# Input:, Output: (seq, 1, 768, 1, 1)
builder.add_add_broadcastable(
	name='embeddings_addition',
	input_names=['token_embeddings', 'positional_embeddings'],
	output_name=f'{0}_previous_block'
)

for i in range(steps):
	print(i)
	ln_weight = model.h[i].ln_1.weight.data.numpy().reshape((1, 1, 768, 1, 1))
	ln_bias = model.h[i].ln_1.bias.data.numpy().reshape((1, 1, 768, 1, 1))
	ln_epsilon = model.h[i].ln_1.eps

	builder.add_mvn(
		name=f"{i}_block_ln_1",
		input_name=f"{i}_previous_block",
		# output_name=f"{i}_block_ln_1_output",
		output_name=f"{i}_block_ln_1",
		across_channels=True,
		normalize_variance=True,
		epsilon=ln_epsilon
	)

	builder.add_scale(
		name=f"{i}_block_ln_1_scaled",
		input_name=f"{i}_block_ln_1",
		output_name=f"{i}_block_ln_1_scaled",
		W=ln_weight,
		b=ln_bias,
		has_bias=True,
		shape_scale=[768],
		shape_bias=[768]
	)

	builder.add_transpose(
		name=f"{i}_block_ln_1_reshape",
		input_name=f"{i}_block_ln_1_scaled",
		output_name=f"{i}_block_ln_1_scaled_transposed",
		axes=(1, 0, 2, 3, 4)
	)


	conv_1D_bias = model.h[i].attn.c_attn.bias.data.numpy().reshape((1, 1, 2304, 1, 1))
	conv_1D_weights = model.h[i].attn.c_attn.weight.data.numpy().transpose().reshape((1, 768, 2304, 1, 1))

	builder.add_inner_product(
		name=f"{i}_block_attn_conv",
		input_name=f"{i}_block_ln_1_scaled_transposed",
		output_name=f"{i}_block_attn_conv",
		input_channels=768,
		output_channels=2304,
		W=conv_1D_weights,
		b=conv_1D_bias,
		has_bias=True
	)

	builder.add_split(
		name=f"{i}_block_attn_qkv_split",
		input_name=f"{i}_block_attn_conv",
		output_names=[f"{i}_block_attn_q", f"{i}_block_attn_k", f"{i}_block_attn_v"]
	)

	builder.add_rank_preserving_reshape(
		name=f"{i}_block_attn_q_reshape",
		input_name=f"{i}_block_attn_q",
		output_name=f"{i}_block_attn_q_reshape",
		output_shape=(1, 1, sequence_length, 12, 64)
	)

	builder.add_transpose(
		name=f"{i}_block_attn_q_reshape_permuted",
		input_name=f"{i}_block_attn_q_reshape",
		output_name=f"{i}_block_attn_q_reshape_permuted",
		axes=(0, 1, 3, 2, 4)
	)

	builder.add_rank_preserving_reshape(
		name=f"{i}_block_attn_k_reshape",
		input_name=f"{i}_block_attn_k",
		output_name=f"{i}_block_attn_k_reshape",
		output_shape=(1, 1, sequence_length, 12, 64)
	)

	builder.add_transpose(
		name=f"{i}_block_attn_k_reshape_permuted",
		input_name=f"{i}_block_attn_k_reshape",
		output_name=f"{i}_block_attn_k_reshape_permuted",
		axes=(0, 1, 3, 4, 2)
	)

	builder.add_rank_preserving_reshape(
		name=f"{i}_block_attn_v_reshape",
		input_name=f"{i}_block_attn_v",
		output_name=f"{i}_block_attn_v_reshape",
		output_shape=(1, 1, sequence_length, 12, 64)
	)

	builder.add_transpose(
		name=f"{i}_block_attn_v_reshape_permuted",
		input_name=f"{i}_block_attn_v_reshape",
		output_name=f"{i}_block_attn_v_reshape_permuted",
		axes=(0, 1, 3, 2, 4)
	)

	builder.add_batched_mat_mul(
		name=f"{i}_block_attn_qv_matmul",
		input_names=[f"{i}_block_attn_q_reshape_permuted", f"{i}_block_attn_k_reshape_permuted"],
		output_name=f"{i}_block_attn_qv_matmul"
	)

	builder.add_scale(
		name=f"{i}_block_attn_qv_matmul_scaled",
		input_name=f"{i}_block_attn_qv_matmul",
		output_name=f"{i}_block_attn_qv_matmul_scaled",
		W=np.array(1/8),
		b=0,
		has_bias=False
	)

	bias_0 = model.h[i].attn.bias
	nd = ns = sequence_length
	b = (model.h[i].attn.bias[:, :, ns-nd:ns, :ns]).unsqueeze(0)

	builder.add_scale(
		name=f"{i}_block_attn_bias",
		input_name=f"{i}_block_attn_qv_matmul_scaled",
		output_name=f"{i}_block_attn_bias",
		W=b,
		b=None,
		has_bias=False,
		shape_scale=[1, sequence_length, sequence_length]
	)

	bias_constant_0 = - 1e4 * (1 - b)

	builder.add_bias(
		name=f"{i}_block_attn_afterbias",
		input_name=f"{i}_block_attn_bias",
		output_name=f"{i}_block_attn_afterbias",
		# output_name=f"output_logits",
		b=bias_constant_0,
		shape_bias=[1, sequence_length, sequence_length],
	)

	builder.add_squeeze(
		name=f"{i}_squeezit",
		input_name=f"{i}_block_attn_afterbias",
		output_name=f"{i}_squeezit",
		axes=[0, 1]
	)

	builder.add_softmax(
		name=f"{i}_block_attn_softmax",
		input_name=f"{i}_squeezit",
		output_name=f"{i}_block_attn_softmax",
	)

	builder.add_expand_dims(
		name=f"{i}_expandit",
		input_name=f"{i}_block_attn_softmax",
		output_name=f"{i}_expandit",
		axes=[0, 1]
	)

	builder.add_batched_mat_mul(
		name=f"{i}_block_full_attention",
		input_names=[f"{i}_expandit", f"{i}_block_attn_v_reshape_permuted"],
		output_name=f"{i}_block_full_attention"
	)

	builder.add_transpose(
		name=f"{i}_block_full_attention_merged_t",
		input_name=f"{i}_block_full_attention",
		output_name=f"{i}_block_full_attention_merged_t",
		axes=[0, 1, 3, 2, 4]
	)

	builder.add_rank_preserving_reshape(
		name=f"{i}_block_full_attention_merged",
		input_name=f"{i}_block_full_attention_merged_t",
		output_name=f"{i}_block_full_attention_merged",
		output_shape=[1, 1, 1, sequence_length, 768]
	)

	builder.add_transpose(
		name=f"{i}_block_attn_conv_proj_t",
		input_name=f"{i}_block_full_attention_merged",
		output_name=f"{i}_block_attn_conv_proj_t",
		axes=[0, 3, 4, 1, 2]
	)

	conv_1D_proj_bias = model.h[i].attn.c_proj.bias.data.numpy().reshape((1, 1, 768, 1, 1))
	conv_1D_proj_weights = model.h[i].attn.c_proj.weight.data.numpy().transpose().reshape((1, 768, 768, 1, 1))

	# Input:, Output: (1, 3, 768, 1, 1)
	builder.add_inner_product(
		name=f"{i}_block_attn_conv_proj",
		input_name=f"{i}_block_attn_conv_proj_t",
		output_name=f"{i}_block_attn_conv_proj",
		input_channels=768,
		output_channels=768,
		W=conv_1D_proj_weights,
		b=conv_1D_proj_bias,
		has_bias=True
	)

	# Input: (seq, 1, 768, 1, 1), Output: (1, seq, 768, 1, 1)
	builder.add_transpose(
		name=f"{i}_previous_block_t",
		input_name=f'{i}_previous_block',
		output_name=f"{i}_previous_block_t",
		axes=[1, 0, 2, 3, 4]
	)

	# Input: [(1, seq, 768, 1, 1), (1, seq, 768, 1, 1)], Output: (1, seq, 768, 1, 1)
	builder.add_add_broadcastable(
		name=f"{i}_block_xa_sum",
		input_names=[f"{i}_previous_block_t", f"{i}_block_attn_conv_proj"],
		output_name=f"{i}_block_xa_sum",
		# output_name=f"output_logits"
	)

	ln_2_weight = model.h[i].ln_2.weight.data.numpy().reshape((1, 1, 768, 1, 1))
	ln_2_bias = model.h[i].ln_2.bias.data.numpy().reshape((1, 1, 768, 1, 1))
	ln_2_epsilon = model.h[i].ln_2.eps

	# Input: (1, seq, 768, 1, 1), Output:
	builder.add_mvn(
		name=f"{i}_block_ln_2",
		input_name=f"{i}_block_xa_sum",
		output_name=f"{i}_block_ln_2",
		across_channels=True,
		normalize_variance=True,
		epsilon=ln_2_epsilon
	)

	builder.add_scale(
		name=f"{i}_block_ln_2_scaled",
		input_name=f"{i}_block_ln_2",
		# output_name=f"output_logits",
		output_name=f"{i}_block_ln_2_scaled",
		W=ln_2_weight,
		b=ln_2_bias,
		has_bias=True,
		shape_scale=[768],
		shape_bias=[768]
	)

	mlp_conv_1D_fc_bias = model.h[i].mlp.c_fc.bias.data.numpy().reshape((1, 1, 3072, 1, 1))
	mlp_conv_1D_fc_weights = model.h[i].mlp.c_fc.weight.data.numpy().transpose().reshape((1, 768, 3072, 1, 1))

	# Input:, Output: (1, 3, 3072, 1, 1)
	builder.add_inner_product(
		name=f"{i}_block_mlp_conv_fc",
		input_name=f"{i}_block_ln_2_scaled",
		output_name=f"{i}_block_mlp_conv_fc",
		# output_name=f"output_logits",
		input_channels=768,
		output_channels=3072,
		W=mlp_conv_1D_fc_weights,
		b=mlp_conv_1D_fc_bias,
		has_bias=True
	)

	builder.add_gelu(
		name=f"{i}_block_mlp_gelu",
		input_name=f"{i}_block_mlp_conv_fc",
		output_name=f"{i}_block_mlp_gelu",
		# output_name=f"output_logits",
		mode='TANH_APPROXIMATION'
	)

	mlp_conv_1D_proj_bias = model.h[i].mlp.c_proj.bias.data.numpy().reshape((1, 1, 768, 1, 1))
	mlp_conv_1D_proj_weights = model.h[i].mlp.c_proj.weight.data.numpy().transpose().reshape((1, 3072, 768, 1, 1))

	# Input:, Output: (1, 3, 3072, 1, 1)
	builder.add_inner_product(
		name=f"{i}_block_mlp_conv_proj",
		input_name=f"{i}_block_mlp_gelu",
		output_name=f"{i}_block_mlp_conv_proj",
		# output_name=f"output_logits",
		input_channels=3072,
		output_channels=768,
		W=mlp_conv_1D_proj_weights,
		b=mlp_conv_1D_proj_bias,
		has_bias=True
	)

	builder.add_add_broadcastable(
		name=f"{i}_block_xm_sum",
		input_names=[f"{i}_block_xa_sum", f"{i}_block_mlp_conv_proj"],
		# output_name=f"output_logits"
		output_name=f"{i + 1}_previous_block_final"
	)

	builder.add_transpose(
		name=f"{i}_block_xm_sum_t",
		input_name=f"{i + 1}_previous_block_final",
		output_name=f"{i + 1}_previous_block",
		axes=[1, 0, 2, 3, 4]
	)


ln_f_weight = model.ln_f.weight.data.numpy().reshape((1, 1, 768, 1, 1))
ln_f_bias = model.ln_f.bias.data.numpy().reshape((1, 1, 768, 1, 1))
ln_f_epsilon = model.ln_f.eps

# Input: (1, seq, 768, 1, 1), Output:
builder.add_mvn(
	name=f"ln_f",
	input_name=f"{steps}_previous_block_final",
	output_name=f"ln_f",
	# output_name=f"output_logits",
	across_channels=True,
	normalize_variance=True,
	epsilon=ln_f_epsilon
)

builder.add_scale(
	name=f"ln_f_scaled",
	input_name=f"ln_f",
	output_name=f"ln_f_scaled",
	# output_name=f"output_logits",
	W=ln_f_weight,
	b=ln_f_bias,
	has_bias=True,
	shape_scale=[768],
	shape_bias=[768]
)

lm_head_weights = lm_head_model.lm_head.weight.data.numpy().reshape((1, 50257, 768, 1, 1))

builder.add_inner_product(
	name="lm_head",
	input_name="ln_f_scaled",
	output_name="output_logits",
	input_channels=768,
	output_channels=50257,
	W=lm_head_weights,
	b=None,
	has_bias=False
)

# compile spec to model
mlmodel = coremltools.models.MLModel(builder.spec)

save_spec(builder.spec, f'../Resources/{model_name}-{sequence_length}-{steps}-2.mlmodel')
# model = coremltools.models.MLModel('gpt2.mlmodel')

# input_ids = np.zeros(sequence_length)
# position_ids = np.arange(sequence_length).astype(np.float)

# input_data = {
# 	'input_ids': input_ids,
# 	'position_ids': position_ids,
# }

# predictions = mlmodel.predict(input_data)["output_logits"]
# equal = np.amax(predictions - mlp_conv_proj.detach().numpy())

# print(predictions)


# save_spec(builder.spec, 'gpt2.mlmodel')
