# MBConv头
MBConv全称: ​Mobile Inverted Bottleneck Convolution, 通过倒置瓶颈结构和深度可分离卷积实现高效特征提取.
基于: MobileNetV2: Inverted Residuals and Linear Bottlenecks.

如下分析代码基于 moat 模型中的 mbconv 实现.

```python
# 代码待分析
class MBConvBlock(tf.keras.layers.Layer):
    def __init__(self, **config):
    self._config = self._retrieve_config(config)
    super().__init__(name=self._config.name)
    self._activation_fn = self._config.activation
    self._norm_class = self._config.norm_class

  def build(self, input_shape: list[int]) -> None:
    input_size = input_shape[-1]
    inner_size = self._config.hidden_size * self._config.expansion_rate

    self._shortcut_conv = None
    if input_size != self._config.hidden_size:
      self._shortcut_conv = tf.keras.layers.Conv2D(
          filters=self._config.hidden_size,
          kernel_size=1,
          strides=1,
          padding='same',
          kernel_initializer=self._config.kernel_initializer,
          bias_initializer=self._config.bias_initializer,
          use_bias=True,
          name='shortcut_conv')

    self._pre_norm = self._norm_class(name='pre_norm')
    self._expand_conv = tf.keras.layers.Conv2D(
        filters=inner_size,
        kernel_size=1,
        strides=1,
        kernel_initializer=self._config.kernel_initializer,
        padding='same',
        use_bias=False,
        name='expand_conv')
    self._expand_norm = self._norm_class(name='expand_norm')
    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=self._config.kernel_size,
        strides=self._config.block_stride,
        depthwise_initializer=self._config.kernel_initializer,
        padding='same',
        use_bias=False,
        name='depthwise_conv')
    self._depthwise_norm = self._norm_class(name='depthwise_norm')

    self._se = None
    if self._config.se_ratio is not None:
      se_filters = max(1, int(self._config.hidden_size * self._config.se_ratio))
      self._se = SqueezeAndExcitation(
          se_filters=se_filters,
          output_filters=inner_size,
          kernel_initializer=self._config.kernel_initializer,
          bias_initializer=self._config.bias_initializer,
          name='se')

    self._shrink_conv = tf.keras.layers.Conv2D(
        filters=self._config.hidden_size,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=self._config.kernel_initializer,
        bias_initializer=self._config.bias_initializer,
        use_bias=True,
        name='shrink_conv')

```

###### call(forward)


```python
def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    shortcut = self._shortcut_branch(inputs)
    output = self._pre_norm(inputs, training=training)
    output = self._expand_conv(output)
    output = self._expand_norm(output, training=training)
    output = self._activation_fn(output)
    output = self._depthwise_conv(output)
    output = self._depthwise_norm(output, training=training)
    output = self._activation_fn(output)
    if self._se:
      output = self._se(output)
    output = self._shrink_conv(output)
    output = residual_add_with_drop_path(
        output, shortcut,
        self._config.survival_prob, training)
    return output
```