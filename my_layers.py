import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Flatten, GlobalAveragePooling1D, Dense, GlobalMaxPooling1D, concatenate, Concatenate


class Attention(Layer):
	def __init__(self, attention_size, **kwargs):
		self.supports_masking = True
		self.attention_size = attention_size
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		# W: (EMBED_SIZE, ATTENTION_SIZE)
		# b: (ATTENTION_SIZE, 1)
		# u: (ATTENTION_SIZE, 1)
		self.W = self.add_weight(name="W_{:s}".format(self.name),
								 shape=(input_shape[-1], self.attention_size),
								 initializer="glorot_normal",
								 trainable=True)
		self.b = self.add_weight(name="b_{:s}".format(self.name),
								 shape=(input_shape[1], 1),
								 initializer="zeros",
								 trainable=True)
		self.u = self.add_weight(name="u_{:s}".format(self.name),
								 shape=(self.attention_size, 1),
								 initializer="glorot_normal",
								 trainable=True)
		super(Attention, self).build(input_shape)

	def call(self, x, mask=None):
		# input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
		# et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
		et = K.tanh(K.dot(x, self.W) + self.b)
		# at: (BATCH_SIZE, MAX_TIMESTEPS)
		at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
		if mask is not None:
			at *= K.cast(mask, K.floatx())
		# ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
		atx = K.expand_dims(at, axis=-1)
		ot = atx * x
		# output: (BATCH_SIZE, EMBED_SIZE)
		output = K.sum(ot, axis=1)
		return output

	def compute_mask(self, input, input_mask=None):
		return None

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])

class MultiHeadAttention(Layer):
	"""多头注意力机制
	"""
	def __init__(self,heads, head_size, output_dim=None, **kwargs):
		self.heads = heads
		self.head_size = head_size
		self.output_dim = output_dim or heads * head_size
		super(MultiHeadAttention, self).__init__(**kwargs)

	def build(self, input_shape):
		# 为该层创建一个可训练的权重
		#inputs.shape = (batch_size, time_steps, seq_len)
		self.kernel = self.add_weight(name='kernel',
									  shape=(3,input_shape[2], self.head_size),
									  initializer='uniform',
									  trainable=True)
		self.dense = self.add_weight(name='dense',
									  shape=(input_shape[2], self.output_dim),
									  initializer='uniform',
									  trainable=True)

		super(MultiHeadAttention, self).build(input_shape)  # 一定要在最后调用它

	def call(self, x):
		out = []
		for i in range(self.heads):
			WQ = K.dot(x, self.kernel[0])
			WK = K.dot(x, self.kernel[1])
			WV = K.dot(x, self.kernel[2])

			# print("WQ.shape",WQ.shape)
			# print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)

			QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
			QK = QK / (100**0.5)
			QK = K.softmax(QK)

			# print("QK.shape",QK.shape)

			V = K.batch_dot(QK,WV)
			out.append(V)
		out = Concatenate(axis=-1)(out)
		out = K.dot(out, self.dense)
		return out

	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[1],self.output_dim)



class GlobalMaxPooling1DWithMasking(GlobalMaxPooling1D):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(GlobalMaxPooling1DWithMasking, self).__init__(**kwargs)

	def compute_mask(self, x, mask):
		return mask

class GlobalAveragePooling1DWithMasking(GlobalAveragePooling1D):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(GlobalAveragePooling1DWithMasking, self).__init__(**kwargs)

	def compute_mask(self, x, mask):
		return mask