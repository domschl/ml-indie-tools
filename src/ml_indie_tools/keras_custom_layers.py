import tensorflow as tf

try:
    # the endless shuffle of keras modules
    import tensorflow.keras as keras
    from tensorflow.keras import layers

    print("Using TF-Keras version:", keras.__version__)
except ImportError:
    import keras
    import keras.layers as layers

    print("Using Keras version:", keras.__version__)
import numpy as np
import math


# sphinx-doc hickup: a member named `call` seems to cause all kinds of sphinx-hickup
# error starting with non-existing line-12 docstrings, if automatic :member: doc
# is activated in index.rst.


class ResidualBlock(layers.Layer):
    """Residual Block layer for Keras

    The residual block consists of two fully connected layers with units neurons
    followed by two BatchNorms and ReLUs:

    .. code-block:: none

        #   ┌──────────────────────────────────────────────────┐
        #   │  ┌─────┐  ┌──┐  ┌────┐    ┌─────┐  ┌──┐  ┌────┐  ▼
        # ──┴─►│Dense│─►│BN│─►│ReLU│───►│Dense│─►│BN│─►│ReLU│─ + ─►    highway=True
        #      └─────┘  └──┘  └────┘    └─────┘  └──┘  └────┘
        #
        #   ┌──────────────────────────────────────────┐
        #   │  ┌─────┐  ┌──┐  ┌────┐    ┌─────┐  ┌──┐  ▼   ┌────┐
        # ──┴─►│Dense│─►│BN│─►│ReLU│───►│Dense│─►│BN│─ + ─►│ReLU│─►    highway=False
        #      └─────┘  └──┘  └────┘    └─────┘  └──┘      └────┘

    The additive residual connection either bridges all layers (highway), or
    connects just before the last ReLU.

    :param units: Positive integer, number of hidden units.
    :param highway: Boolean, whether to use highway connection or not.
    """

    def __init__(self, units, highway=False, **kwargs):
        self.units = units
        self.highway = highway
        super(ResidualBlock, self).__init__(**kwargs)
        self.dense1 = layers.Dense(self.units)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dense2 = layers.Dense(self.units)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "highway": self.highway})
        return config

    def call(
        self, inputs
    ):  # This member name kills sphinx's autodoc for members! Beware!
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.bn2(x)
        if self.highway:
            x = self.relu2(x)
            x = x + inputs
        else:
            x = x + inputs
            x = self.relu2(x)
        return x


class ResidualDense(layers.Layer):
    """Residual Dense layer for Keras

    The residual dense layer consists of a fully connected layer followed by BatchNorm and ReLU:

    .. code-block:: none

        #   ┌─────────────────────────┐
        #   │  ┌─────┐  ┌──┐  ┌────┐  ▼
        # ──┴─►│Dense│─►│BN│─►│ReLU│─ + ─►
        #      └─────┘  └──┘  └────┘

    :param units: Positive integer, number of hidden units.
    :param regularizer: Positive float, regularization strength for the Dense layer.
    """

    def __init__(self, units, regularizer=0, **kwargs):
        self.units = units
        self.regularizer = regularizer
        super(ResidualDense, self).__init__(**kwargs)
        if self.regularizer != 0:
            self.dense1 = layers.Dense(
                self.units, kernel_regularizer=keras.regularizers.l2(self.regularizer)
            )
        else:
            self.dense1 = layers.Dense(self.units)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "regularizer": self.regularizer})
        return config

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.relu(x)
        x = self.bn1(x)
        x = x + inputs
        return x


class ResidualDenseStack(layers.Layer):
    """Residual Dense layer for Keras

    The residual dense layer stack consists of `layer_count` :class:`ResidualDense` layers.

    .. code-block:: none

        #  ┌─────────── n ─────────────┐   n = layer_count repetitions
        #   ┌─────────────────────────┐
        #   │  ┌─────┐  ┌──┐  ┌────┐  ▼
        # ──┴─►│Dense│─►│BN│─►│ReLU│─ + ─►
        #      └─────┘  └──┘  └────┘

    :param units: Positive integer, number of hidden units.
    :param layer_count: Positive integer, number of layer-blocks, each a `ResidualDense` block.
    :param regularizer: Positive float, regularization strength for the Dense layer.
    """

    def __init__(self, units, layer_count, regularizer=0, **kwargs):
        self.units = units
        self.layer_count = layer_count
        self.regularizer = regularizer

        super(ResidualDenseStack, self).__init__(**kwargs)
        self.rd = []
        for _ in range(0, self.layer_count):
            self.rd.append(ResidualDense(self.units, regularizer=self.regularizer))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "layers": self.layer_count,
                "regularizer": self.regularizer,
            }
        )
        return config

    def call(self, inputs):
        x = self.rd[0](inputs)
        for i in range(1, self.layer_count):
            x = self.rd[i](x)
        return x


class ParallelResidualDenseStacks(layers.Layer):
    """Parallel Residual Dense Stacks layer for Keras

    The parallel residual dense layer stacks consist of `stacks` count parallel
    :class:`ResidualDenseStack`, each of which consists of  `layer_count` :class:`ResidualDense`
    layers. The output of all parallel stacks is concatenated and scaled down to `units` units.

    .. code-block:: none

        #        ┌─────────── n ─────────────┐   n = layer_count repetitions
        #         ┌─────────────────────────┐
        #         │  ┌─────┐  ┌──┐  ┌────┐  ▼    ┌──────┐
        #   ┌─────┴─►│Dense│─►│BN│─►│ReLU│─ + ─► │      │
        #   │        └─────┘  └──┘  └────┘       │      │
        #   │                                    │      │
        #   │    ┌─────────── n ─────────────┐   │      │
        #   │     ┌─────────────────────────┐    │      │
        #   │     │  ┌─────┐  ┌──┐  ┌────┐  ▼    │concat│   ┌─────┐  ┌────┐
        #   ├─────┴─►│Dense│─►│BN│─►│ReLU│─ + ─► │      │ ─►│Dense│─►│ReLU│─►
        # ──┤        └─────┘  └──┘  └────┘       │      │   └─────┘  └────┘
        #   │                .                   │      │    scale down to
        #   │                . `stacks` reps     │      │    `units`.
        #   │                .                   │      │
        #   │    ┌─────────── n ─────────────┐   │      │
        #   │     ┌─────────────────────────┐    │      │
        #   │     │  ┌─────┐  ┌──┐  ┌────┐  ▼    │      │
        #   └─────┴─►│Dense│─►│BN│─►│ReLU│─ + ─► │      │
        #            └─────┘  └──┘  └────┘       └──────┘

    :param units: Positive integer, number of hidden units.
    :param layer_count: Positive integer, number of layer-blocks, each a `ResidualDense` block.
    :param stacks: Positive integer, number of parallel stacks.
    :param regularizer: Positive float, regularization strength for the Dense layer.
    """

    def __init__(self, units, layer_count, stacks, dispatch, regularizer=0, **kwargs):
        super(ParallelResidualDenseStacks, self).__init__(**kwargs)
        self.units = units
        self.layer_count = layer_count
        self.stacks = stacks
        self.dispatch = dispatch
        self.regularizer = regularizer

        if self.dispatch is True:
            self.scale = layers.Dense(units * stacks, activation=None)
        else:
            self.scale = layers.Dense(units, activation=None)

        self.rds = []
        for _ in range(0, self.stacks):
            self.rds.append(
                ResidualDenseStack(
                    self.units, self.layer_count, regularizer=self.regularizer
                )
            )
        self.rescale_relu = layers.ReLU()
        self.concat = layers.Concatenate()
        if self.regularizer != 0:
            self.rescale = layers.Dense(
                self.units, kernel_regularizer=keras.regularizers.l2(self.regularizer)
            )
        else:
            self.rescale = layers.Dense(self.units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "layers": self.layer_count,
                "stacks": self.stacks,
                "dispatch": self.dispatch,
                "regularizer": self.regularizer,
            }
        )
        return config

    def call(self, inputs):
        xa = []
        # Scale up
        x = self.scale(inputs)
        for i in range(0, self.stacks):
            if self.dispatch:
                xa.append(self.rds[i](x[:, i * self.units : (i + 1) * self.units]))
            else:
                xa.append(self.rds[i](x))
        x = self.concat(xa)
        x = self.rescale(x)
        x = self.rescale_relu(x)
        return x


class SelfAttention(layers.Layer):
    """Self-attention layer for Keras

    The self-attention layer learns three matrices (key :math:`W_k`, query :math:`W_q`, value :math:`W_v`)
    that provide context-information for the :math:`input`.
    Input is mutiplied with all three matrices, then :math:`W_k` and :math:`W_q` are multiplied,
    scaled down by :math:`\\sqrt{\\dim{input}[-1]}` and normalized, either by LayerNorm,
    BatchNorm or Softmax or not at all. The result is then multiplied with :math:`W_v`, and, if hidden
    dimension of the :math:`W_{x_i}` matrices is different from input units last dimension,
    rescaled by a final dense matrix multiply. Output has same shape as input.

    .. code-block:: none

        #
        #     ┌──┐
        #  ┌► │Wk│───┐   ┌─────┐
        #  │  └──┘   │   │Scale│
        #  │  ┌──┐   × ─►│Norm │─┐   (opt.)
        # ─┼─►│Wq│───┘   └─────┘ │   ┌─────┐
        #  │  └──┘               │   │Scale│──►
        #  │  ┌──┐               × ─►│Dense│
        #  └► │Wv│───────────────┘   └─────┘
        #     └──┘
        #

    :param units: Positive integer, number of hidden units. The matrices :math:`W_{x_i}` are of shape :math:`hs \\times hs`.
    :param norm: either 'batchnorm', 'layernorm', 'softmax', or None
    """

    def __init__(self, units=None, norm=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        self.norm = norm
        if self.norm == "layernorm":
            self.norm = layers.LayerNormalization(axis=-1)
        elif self.norm == "batchnorm":
            self.norm = layers.BatchNormalization()
        elif self.norm == "softmax":
            self.norm = layers.Softmax()
        elif self.norm is None or self.norm == "none":
            self.norm = None
        else:
            raise ValueError("Unknown norm: {}".format(self.norm))
        self.pm = layers.Permute((2, 1))

    def build(self, input_shape):
        self.fact = math.sqrt(input_shape[-1])
        if self.units is None:
            dim2 = input_shape[-1]
        else:
            dim2 = self.units
            self.scale = self.add_weight(
                shape=(dim2, input_shape[-1]),
                initializer="random_normal",
                name="w1",
                trainable=True,
            )
        self.w_keys = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w2",
            trainable=True,
        )
        self.w_queries = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w3",
            trainable=True,
        )
        self.w_values = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w4",
            trainable=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "norm": self.norm})
        return config

    def call(self, inputs):
        vk = tf.matmul(inputs, self.w_keys)
        vq = tf.matmul(inputs, self.w_queries)
        vv = tf.matmul(inputs, self.w_values)
        kq = tf.matmul(vk, vq, transpose_b=True)
        kqs = kq / self.fact
        if self.norm is not None:
            sn = self.norm(kqs)
        else:
            sn = kqs
        out = tf.matmul(sn, self.pm(vv), transpose_b=True)

        if self.units is not None:
            out = tf.matmul(out, self.scale)
        return out


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention layer for Keras

    The multi-head self-attention layer concatenates the output of `heads` :class:`SelfAttention`
    layers. Each of the self-attention layers has an additive residual connection.
    If `mh_normalize` is True, the concatenated output is normalized.
    After scaling down to the number of units, the output is then passed through a
    ReLU and Dense layer again with residual connection.
    Finally, optional normalization and a final optional ReLU is applied.
    Output has same shape as input.

    .. code-block:: none

        #    ┌──────────────┐
        #    │  ┌────────┐  ▼   ┌──────┐  ┌────┐
        #  ┌─┴─►│SelfAtt.│─ + ─►│      │  │    │
        #  │    └────────┘      │      │  │    │
        #  │ ┌──────────────┐   │      │  │    │          ┌───────────────────┐   ┌────┐
        # ─┤ │  ┌────────┐  ▼   │      │  │Opt.│  ┌─────┐ │  ┌────┐  ┌─────┐  ▼   │Opt │
        #  ├─┴─►│SelfAtt.│─ + ─►│      │─►│Norm│─►│Scale│─┴─►│ReLU│─►│Dense│─ + ─►│Norm│─►
        #  │    └────────┘      │concat│  │    │  └─────┘    └────┘  └─────┘      └────┘
        #  │        .           │ or   │  │    │
        #  │        . head      │ relu │  │    │
        #  │        . reps      │ +add │  │    │
        #  │ ┌──────────────┐   │      │  │    │
        #  │ │  ┌────────┐  ▼   │      │  │    │
        #  └─┴─►│SelfAtt.│─ + ─►│      │  │    │
        #       └────────┘      └──────┘  └────┘

    :param units: Positive integer `hs`, number of hidden units.
    :param heads: Positive integer, number of self-attention heads.
    :param mh_normalize: Boolean, whether to normalize the output of the multi-head self-attention.
    :param norm: either 'batchnorm', 'layernorm, or 'softmax', the normalization used within each self-attention head.
    :param final_relu: Boolean, whether to apply a ReLU after the final scaling and dense layer.
    :param join_heads_by_add: on true heads are added after additional relu-nonlin, instead of concatenated (original all-you-need).
    :param recurrent: Boolean, whether to use a recurrent self-attention layer.
    :param gated_memory: Boolean, whether to use a gated memory self-attention layer.
    """

    def __init__(
        self,
        heads,
        units=None,
        norm=None,
        mh_normalize=True,
        final_relu=False,
        join_heads_by_add=False,
        recurrent=False,
        gated_memory=False,
        **kwargs
    ):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.units = units
        self.norm = norm
        self.mh_normalize = mh_normalize
        self.final_relu = final_relu
        self.recurrent = recurrent
        self.gated_memory = gated_memory
        if self.gated_memory is True:
            self.recurrent = False
        self.mhsa = []
        for _ in range(0, self.heads):
            if self.gated_memory is True:
                self.mhsa.append(
                    GatedMemorySelfAttention(units=self.units, norm=self.norm)
                )
            elif recurrent is True:
                self.mhsa.append(
                    RecurrentSelfAttention(units=self.units, norm=self.norm)
                )
            else:
                self.mhsa.append(SelfAttention(units=self.units, norm=self.norm))
        self.join_heads_by_add = join_heads_by_add
        if self.join_heads_by_add is False:
            self.cc = layers.Concatenate(axis=1)
        if self.mh_normalize is True:
            self.ln1 = layers.LayerNormalization()
            self.ln2 = layers.LayerNormalization()
        self.relu1 = layers.ReLU()
        self.relu2 = layers.ReLU()
        self.pm = layers.Permute((2, 1))

    def build(self, input_shape):
        if self.join_heads_by_add is False:
            self.w_heads = self.add_weight(
                shape=(self.heads * input_shape[-1], input_shape[-1]),
                initializer="random_normal",
                name="w5concat",
                trainable=True,
            )
        else:
            self.w_heads = self.add_weight(
                shape=(input_shape[-1], input_shape[-1]),
                initializer="random_normal",
                name="w5add",
                trainable=True,
            )
        self.lin = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="random_normal",
            name="w6",
            trainable=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "heads": self.heads,
                "units": self.units,
                "norm": self.norm,
                "mh_normalize": self.mh_normalize,
                "final_relu": self.final_relu,
                "recurrent": self.recurrent,
                "gated_memory": self.gated_memory,
                "join_heads_by_add": self.join_heads_by_add,
            }
        )
        return config

    def call(self, inputs, memory=None):
        xa = []
        if memory is None:
            mem = tf.zeros_like(inputs)
        else:
            mem = memory
        for i in range(0, self.heads):
            if self.recurrent is True:
                xai, mem = self.mhsa[i](inputs, mem)
            else:
                xai = self.mhsa[i](inputs)
            xa.append(self.pm(xai + inputs))
        if self.join_heads_by_add is True:
            for i in range(len(xa)):
                if i == 0:
                    x = self.relu2(xa[i])
                else:
                    x = x + self.relu2(xa[i])
            x = self.pm(x)
        else:
            x = self.pm(self.cc(xa))
        if self.mh_normalize is True:
            x = self.ln1(x)
        xt = tf.matmul(x, self.w_heads)
        x = self.relu1(xt)
        x = tf.matmul(x, self.lin) + xt
        if self.mh_normalize is True:
            x = self.ln2(x)
        if self.recurrent is True and memory is not None:
            return x, mem
        else:
            return x


class PositionalEncoding(layers.Layer):
    """Positional encoding layer.

    adds sinusoid of different frequencies to the input. Can be used to add sequence-information to input
    data for transformers or attention layers.

    :param amplitude: float, amplitude of the encoding, default=1.0.
    :param trainable: boolean, whether the weights of the layer are trainable, default=False.
    """

    def __init__(self, amplitude=1.0, trainable=False, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.amplitude = amplitude
        self.trainable = trainable

    # positional encoding taken from: https://www.tensorflow.org/text/tutorials/transformer
    @staticmethod
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self, position, d_model):
        angle_rads = PositionalEncoding._get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...] * self.amplitude
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "amplitude": self.amplitude,
                "trainable": self.trainable,
            }
        )
        return config

    def build(self, input_shape):
        self.pe = self._positional_encoding(input_shape[1], input_shape[2])

    def call(self, inputs):
        return tf.add(inputs, self.pe)


class RecurrentSelfAttention(layers.Layer):
    """Self-attention layer for Keras

    The self-attention layer learns three matrices (key :math:`W_k`, query :math:`W_q`, value :math:`W_v`)
    that provide context-information for the :math:`input`. Additionally, recurrent state is maintained
    by :math:`W_{memory}`.
    Input is mutiplied with all three matrices, then :math:`W_k` and :math:`W_q` are multiplied,
    scaled down by :math:`\\sqrt{\\dim{input}[-1]}` and normalized, either by LayerNorm,
    BatchNorm or Softmax or not at all. The result is then multiplied with :math:`W_v`, and, if hidden
    dimension of the :math:`W_{x_i}` matrices is different from input units last dimension,
    rescaled by a final dense matrix multiply. Output has same shape as input.

    .. code-block:: none

        #
        #     ┌──────────┐
        #  ┌► │Wk+Wmemory│───┐   ┌─────┐
        #  │  └──────────┘   │   │Scale│
        #  │  ┌──┐           × ─►│Norm │─┐   (opt.)
        # ─┼─►│Wq│───────────┘   └─────┘ │   ┌─────┐
        #  │  └──┘                       │   │Scale│─────────+────────► output
        #  │  ┌──┐                       × ─►│Dense│         +
        #  └► │Wv│───────────────────────┘   └─────┘       Wmemory ───► memory
        #     └──┘
        #

    :param units: Positive integer, number of hidden units. The matrices :math:`W_{x_i}` are of shape :math:`hs \\times hs`.
    :param norm: either 'batchnorm', 'layernorm', 'softmax', or None
    """

    def __init__(self, units=None, norm=None, **kwargs):
        super(RecurrentSelfAttention, self).__init__(**kwargs)
        self.units = units
        self.norm = norm
        if self.norm == "layernorm":
            self.norm = layers.LayerNormalization(axis=-1)
        elif self.norm == "batchnorm":
            self.norm = layers.BatchNormalization()
        elif self.norm == "softmax":
            self.norm = layers.Softmax()
        elif self.norm is None or self.norm == "none":
            self.norm = None
        else:
            raise ValueError("Unknown norm: {}".format(self.norm))
        self.pm = layers.Permute((2, 1))

    def build(self, input_shape):
        self.fact = math.sqrt(input_shape[-1])
        if self.units is None or self.units == input_shape[-1]:
            dim2 = input_shape[-1]
            self.reshape = False
        else:
            dim2 = self.units
            self.scale = self.add_weight(
                shape=(dim2, input_shape[-1]),
                initializer="random_normal",
                name="w1",
                trainable=True,
            )
            self.reshape = True
        self.w_keys = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w2",
            trainable=True,
        )
        self.w_queries = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w3",
            trainable=True,
        )
        self.w_values = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w4",
            trainable=True,
        )
        self.w_memory = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="zeros",
            name="w5",
            trainable=False,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "norm": self.norm})
        return config

    def call(self, inputs, memory=None):
        if memory is None or self.reshape is True:
            vk = tf.matmul(inputs, self.w_keys)
        else:
            vm = tf.matmul(memory, self.w_memory)
            vm = tf.tanh(vm)
            vk = tf.matmul(inputs, self.w_keys) + vm
        vq = tf.matmul(inputs, self.w_queries)
        vv = tf.matmul(inputs, self.w_values)
        kq = tf.matmul(vk, vq, transpose_b=True)
        kqs = kq / self.fact
        if self.norm is not None:
            sn = self.norm(kqs)
        else:
            sn = kqs
        out = tf.matmul(sn, self.pm(vv), transpose_b=True)

        if self.reshape is True:
            out = tf.matmul(out, self.scale)
        if memory is None or self.reshape is True:
            return out, memory
        else:
            mv = tf.matmul(inputs, self.w_memory)
            mv = tf.tanh(mv)
            mvq = tf.matmul(mv, vm, transpose_b=True)
            ml = tf.matmul(mvq, self.pm(vv), transpose_b=True)
            self.w_memory = ml
            return out, self.w_memory


class GatedMemorySelfAttention(layers.Layer):
    """Self-attention layer for Keras

    The self-attention layer learns three matrices (key :math:`W_k`, query :math:`W_q`, value :math:`W_v`)
    that provide context-information for the :math:`input`. Additionally, a gated exponentially decaying
    memory state is maintained.
    Input is mutiplied with all three matrices, then :math:`W_k` and :math:`W_q` are multiplied,
    scaled down by :math:`\\sqrt{\\dim{input}[-1]}` and normalized, either by LayerNorm,
    BatchNorm or Softmax or not at all. The result is then multiplied with :math:`W_v`, and, if hidden
    dimension of the :math:`W_{x_i}` matrices is different from input units last dimension,
    rescaled by a final dense matrix multiply. Output has same shape as input.

    .. code-block:: none

        #  XXX memory missing
        #     ┌──┐
        #  ┌► │Wk│───┐   ┌─────┐
        #  │  └──┘   │   │Scale│
        #  │  ┌──┐   × ─►│Norm │─┐   (opt.)
        # ─┼─►│Wq│───┘   └─────┘ │   ┌─────┐
        #  │  └──┘               │   │Scale│──────────► output
        #  │  ┌──┐               × ─►│Dense│
        #  └► │Wv│───────────────┘   └─────┘
        #     └──┘
        #

    :param units: Positive integer, number of hidden units. The matrices :math:`W_{x_i}` are of shape :math:`hs \\times hs`.
    :param norm: either 'batchnorm', 'layernorm', 'softmax', or None
    """

    def __init__(self, units=None, norm=None, **kwargs):
        super(GatedMemorySelfAttention, self).__init__(**kwargs)
        self.units = units
        self.norm = norm
        if self.norm == "layernorm":
            self.norm = layers.LayerNormalization(axis=-1)
        elif self.norm == "batchnorm":
            self.norm = layers.BatchNormalization()
        elif self.norm == "softmax":
            self.norm = layers.Softmax()
        elif self.norm is None or self.norm == "none":
            self.norm = None
        else:
            raise ValueError("Unknown norm: {}".format(self.norm))
        self.pm = layers.Permute((2, 1))
        self.retain_factor = 0.8

    def build(self, input_shape):
        self.fact = math.sqrt(input_shape[-1])
        if self.units is None or self.units == input_shape[-1]:
            dim2 = input_shape[-1]
            self.reshape = False
        else:
            dim2 = self.units
            self.scale = self.add_weight(
                shape=(dim2, input_shape[-1]),
                initializer="random_normal",
                name="w1",
                trainable=True,
            )
            self.reshape = True
        self.w_keys = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w2",
            trainable=True,
        )
        self.w_queries = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w3",
            trainable=True,
        )
        self.w_values = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w4",
            trainable=True,
        )
        self.w_memory_gate = self.add_weight(
            shape=(input_shape[-1], dim2),
            initializer="random_normal",
            name="w5",
            trainable=not self.reshape,
        )
        self.w_input_exp_memory = self.add_weight(
            shape=(dim2, input_shape[-1]),
            initializer="zeros",
            name="w6",
            trainable=False,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "norm": self.norm})
        return config

    def call(self, inputs):
        # self.w_input_exp_memory = tf.math.multiply(
        #     self.w_input_exp_memory, self.retain_factor
        # ) + tf.math.multiply(inputs, 1.0 - self.retain_factor)
        vk = tf.matmul(inputs, self.w_keys)
        vq = tf.matmul(inputs, self.w_queries)
        if self.reshape is False:
            vm = tf.transpose(tf.matmul(inputs, self.w_memory_gate), perm=[0, 2, 1])
            memory = tf.tanh(tf.reduce_mean(vm, axis=0, keepdims=False))
            self.w_input_exp_memory = tf.math.multiply(
                self.w_input_exp_memory, self.retain_factor
            ) + tf.math.multiply(memory, 1.0 - self.retain_factor)

            vk = tf.math.multiply(vk, self.retain_factor) + tf.math.multiply(
                self.w_input_exp_memory, 1.0 - self.retain_factor
            )
        vv = tf.matmul(inputs, self.w_values)
        kq = tf.matmul(vk, vq, transpose_b=True)
        kqs = kq / self.fact
        if self.norm is not None:
            sn = self.norm(kqs)
        else:
            sn = kqs
        out = tf.matmul(sn, self.pm(vv), transpose_b=True)

        if self.reshape is True:
            out = tf.matmul(out, self.scale)
        return out
