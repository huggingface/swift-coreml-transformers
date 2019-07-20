import coremltools
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
from coremltools.models.utils import save_spec
import numpy as np

# get weights
from pytorch_transformers import GPT2Model
model = GPT2Model.from_pretrained("gpt2")
wte = model.wte.weight.data.numpy().transpose() # shape (768, 50257) /!\ i hate this
wpe = model.wpe.weight.data.numpy().transpose() # shape (768, 1024)


# build model

input_features = [
	('input_ids', datatypes.Array(512)),
	('position_ids', datatypes.Array(512)),
]
output_features = [ ('output_logits', None) ]

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
builder.add_add_broadcastable(
	name='embeddings_addition',
	input_names=['token_embeddings', 'positional_embeddings'],
	output_name='output_logits'
)

# compile spec to model
## model = coremltools.models.MLModel(builder.spec)
save_spec(builder.spec, 'gpt2.mlmodel')
