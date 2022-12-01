# # import mindspore
# # def positional_encoding(
# #         pos_in, freq=32, include_input=True, log_sampling=True
# # ) -> mindspore.Tensor:
# #     # Whether or not include the input in positional encoding
# #     pos_out = [pos_in] if include_input else []
# #
# #     # Shape of freq_bands: (freq)
# #     if log_sampling:
# #         # freq_bands = 2.0 ** torch.linspace(0.0, freq - 1, freq).to(pos_in)
# #         freq_bands = 2.0 ** mindspore.ops.LinSpace(0.0, freq - 1, freq).to(pos_in)
# #     else:
# #         freq_bands = mindspore.ops.LinSpace(2.0 ** 0.0, 2.0 ** (freq - 1), freq).to(pos_in)
# #         # freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (freq - 1), freq).to(pos_in)
# #
# #     # TODO: why reduce \pi when calculating sin and cos
# #     for freq in freq_bands:
# #         for func in [mindspore.ops.Sin, mindspore.ops.Cos]:
# #             # for func in [torch.sin, torch.cos]:
# #             pos_out.append(func(pos_in * freq))
# #
# #     # pos_out = torch.cat(pos_out, dim=-1)
# #     return pos_out
# # pos_in = mindspore.ops.ones((3),mindspore.float32)
# # freq = mindspore.Tensor(32,mindspore.float32)
# # include_input = True
# # log_sampling = True
# # positional_encoding(pos_in,freq)
#
# def positional_encoding(
#         pos_in, freq=32, include_input=True, log_sampling=True
# ) -> mindspore.Tensor:
#     # torch.Tensor
#     r"""Apply positional encoding to the input. (Section 5.1 of original paper)
#     We use positional encoding to map continuous input coordinates into a
#     higher dimensional space to enable our MLP to more easily approximate a
#     higher frequency function.
#
#     Args:
#       pos_in: input tensor to be positionally encoded, (H*W*num_samples, 3) for sampled point
#       freq: mapping from R into a higher dimensional space R^(2L), in which
#                  L is called the frequency
#       include_input: whether or not to include the input in positional encoding
#       log_sampling: sample logarithmically in frequency space, otherwise linearly
#
#     Returns:
#       pos_out: positional encoding of the input tensor.
#                (H*W*num_samples, (include_input + 2*freq) * 3)
#     """
#     # Whether or not include the input in positional encoding
#     pos_out = [pos_in] if include_input else []
#
#     # Shape of freq_bands: (freq)
#     if log_sampling:
#         # freq_bands = 2.0 ** torch.linspace(0.0, freq - 1, freq).to(pos_in)
#         freq_bands = 2.0 ** mindspore.ops.LinSpace(0.0, freq - 1, freq).to(pos_in)
#     else:
#         freq_bands = mindspore.ops.LinSpace(2.0 ** 0.0, 2.0 ** (freq - 1), freq).to(pos_in)
#         # freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (freq - 1), freq).to(pos_in)
#
#     # TODO: why reduce \pi when calculating sin and cos
#     for freq in freq_bands:
#         for func in [mindspore.ops.Sin, mindspore.ops.Cos]:
#             # for func in [torch.sin, torch.cos]:
#             pos_out.append(func(pos_in * freq))
#
#     pos_out = torch.cat(pos_out, dim=-1)
#     return pos_out

