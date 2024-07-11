from utils.observation_utils import function_wrapper, lower_linalg_to_loops
from random import randint, choice, shuffle
from tqdm import tqdm
import json, re, multiprocessing, os
from copy import copy

BATCH_SIZES = [4, 8, 16, 32, 64, 256]
SIZES = [2**power for power in range(5, 11)]
HEIGHTS = [2**power for power in range(5, 10)]
# HEIGHTS = [2**power for power in range(8, 13)]
CHANNELS = [2**power for power in range(3, 11)]
KERNELS = [1, 3, 5, 7]
DILATIONS = [1, 2, 3]
STRIDES = [1, 2, 3]
SMALL, MEDIUM, BIG = 250, 500, 1000

def choice_topped(choices, max_value):
    trials_left = 50
    n = choice(choices)
    while not (n <= max_value) and trials_left != 0:
        n = choice(choices)
        trials_left -= 1

    if trials_left == 0:
        return None
    return n


def add():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.add ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def sub():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.sub ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def max():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.max ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def mul():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.mul ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def abs():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.abs ins(%arg0: tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def ceil():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.ceil ins(%arg0 : tensor<{SHAPE}xf32>) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def copy_():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.copy ins(%arg0 : tensor<{SHAPE}xf32>) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def fill():
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(randint(1, 4))])
    return f"linalg.fill ins(%arg0 : f32) outs(%arg1: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"


def transpose():
    L = randint(1, 5)

    permutation = list(range(L))
    shuffle(permutation)

    SHAPE1 = [choice(HEIGHTS) for _ in range(L)]

    SHAPE2 = []
    for i in range(L):
        SHAPE2.append(SHAPE1[permutation[i]])

    SHAPE1 = "x".join(map(str, SHAPE1))
    SHAPE2 = "x".join(map(str, SHAPE2))

    return f"linalg.transpose ins(%input:tensor<{SHAPE1}xf32>) outs(%init:tensor<{SHAPE2}xf32>) permutation = {permutation}"


def batch_matmul():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.batch_matmul ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2 : tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_matmul_transpose_a():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.batch_matmul_transpose_a ins(%arg0, %arg1: tensor<{B}x{K}x{N}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2: tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_matmul_transpose_b():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{M}x{K}xf32>) outs(%arg2: tensor<{B}x{N}x{M}xf32>) -> tensor<{B}x{N}x{M}xf32>"


def batch_reduce_matmul():
    B = choice(BATCH_SIZES)
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<{B}x{N}x{K}xf32>, tensor<{B}x{K}x{M}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul():
    N = choice(SIZES)
    K = choice(SIZES)
    M = choice(SIZES)
    return f"linalg.matmul ins(%arg0, %arg1 : tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%arg2 : tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul_transpose_a():
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.matmul_transpose_a ins(%arg0, %arg1: tensor<{K}x{N}xf32>, tensor<{K}x{M}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def matmul_transpose_b():
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
    return f"linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<{N}x{K}xf32>, tensor<{M}x{K}xf32>) outs(%arg2: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>"


def conv_1d():
    N = choice(HEIGHTS)
    F = choice_topped(KERNELS, N)
    N_ = N - F + 1
    return f"linalg.conv_1d ins(%input, %filter : tensor<{N}xf32>, tensor<{F}xf32>) outs(%output : tensor<{N_}xf32>) -> tensor<{N_}xf32>"


def conv_1d_ncw_fcw():
    # INPUT: NCW1
    # KERNL: FCW2
    # OUTPUT: (N, F, W1-W2+1)

    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W1 = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    W2 = choice_topped(KERNELS, (W1 + 2 * padding - 1) // dilation - 1)

    W3 = ((W1 + 2 * padding - dilation * (W2 - 1) - 1) // stride) + 1

    return f"linalg.conv_1d_ncw_fcw {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W1}xf32>, tensor<{F}x{C}x{W2}xf32>) outs (%init: tensor<{N}x{F}x{W3}xf32>) -> tensor<{N}x{F}x{W3}xf32>"


def conv_1d_nwc_wcf():
    # INPUT: NWC
    # KERNL: WCF
    # OUTPUT: (N, W1-W2+1, F)

    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W1 = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    W2 = choice_topped(KERNELS, (W1 + 2 * padding - 1) // dilation - 1)

    W3 = ((W1 + 2 * padding - dilation * (W2 - 1) - 1) // stride) + 1

    return f"linalg.conv_1d_nwc_wcf {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W1}x{C}xf32>, tensor<{W2}x{C}x{F}xf32>) outs (%init: tensor<{N}x{W3}x{F}xf32>) -> tensor<{N}x{W3}x{F}xf32>"


def conv_2d():
    H, W = choice(HEIGHTS), choice(HEIGHTS)

    F1 = F2 = choice_topped(KERNELS, min(H - 2, W - 2))

    H_ = H - F1 + 1
    W_ = W - F2 + 1

    return f"linalg.conv_2d ins(%input, %filter: tensor<{H}x{W}xi32>, tensor<{F1}x{F2}xi32>) outs(%output: tensor<{H_}x{W_}xi32>) -> tensor<{H_}x{W_}xi32>"


def conv_2d_nchw_fchw():
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nchw_fchw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{F}x{C}x{KH}x{KW}xf32>) outs (%init: tensor<{N}x{F}x{H_}x{W_}xf32>) -> tensor<{N}x{F}x{H_}x{W_}xf32>"


def conv_2d_ngchw_fgchw():
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    N = choice(BATCH_SIZES)
    G = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_ngchw_fgchw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{G}x{C}x{H}x{W}xf32>, tensor<{G}x{F}x{C}x{KH}x{KW}xf32>) outs (%init: tensor<{N}x{G}x{F}x{H_}x{W_}xf32>) -> tensor<{N}x{G}x{F}x{H_}x{W_}xf32>"


def conv_2d_nhwc_fhwc():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nhwc_fhwc {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{F}x{KH}x{KW}x{C}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{F}xf32>) -> tensor<{N}x{H_}x{W_}x{F}xf32>"


def conv_2d_nhwc_hwcf():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

    return f"linalg.conv_2d_nhwc_hwcf {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{KH}x{KW}x{C}x{F}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{F}xf32>) -> tensor<{N}x{H_}x{W_}x{F}xf32>"


def conv_3d():
    H, W, D = choice(HEIGHTS), choice(HEIGHTS), choice(HEIGHTS)

    F = choice_topped(KERNELS, min(H, W, D) - 2)

    H_ = H - F + 1
    W_ = W - F + 1
    D_ = D - F + 1

    return f"linalg.conv_3d ins(%input, %filter: tensor<{H}x{W}x{D}xf32>, tensor<{F}x{F}x{F}xf32>) outs(%output: tensor<{H_}x{W_}x{D_}xf32>) -> tensor<{H_}x{W_}x{D_}xf32>"


def conv_3d_ncdhw_fcdhw():
    # INPUT: NCHW
    # KERNL: FCHW
    # OUTPUT: (N, F, H', W')

    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)
    D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    F = choice(CHANNELS)
    KH = KW = KD = choice_topped(
        KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1
    )

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (KD - 1) - 1) // stride) + 1

    return f"linalg.conv_3d_ncdhw_fcdhw {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}x{D}xf32>, tensor<{F}x{C}x{KH}x{KW}x{KD}xf32>) outs (%init: tensor<{N}x{F}x{H_}x{W_}x{D_}xf32>) -> tensor<{N}x{F}x{H_}x{W_}x{D_}xf32>"


def depthwise_conv_1d_ncw_cw():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_ncw_cw {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{C}x{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def depthwise_conv_1d_nwc_wc():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_nwc_wc {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}x{C}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def depthwise_conv_1d_nwc_wcm():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    M = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (W + 2 * padding - 1) // dilation - 1)

    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_1d_nwc_wcm {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{W_}x{C}x{M}xf32>"


def depthwise_conv_2d_nchw_chw():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nchw_chw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{C}x{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


def depthwise_conv_2d_nhwc_hwc():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nhwc_hwc {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{C}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def depthwise_conv_2d_nhwc_hwcm():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    M = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_2d_nhwc_hwcm {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{H_}x{W_}x{C}x{M}xf32>"


def depthwise_conv_3d_ncdhw_cdhw():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)
    D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ncdhw_cdhw {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{C}x{D}x{H}x{W}xf32>, tensor<{C}x{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{D_}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{D_}x{H_}x{W_}xf32>"


def depthwise_conv_3d_ndhwc_dhwc():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)
    D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ndhwc_dhwc {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}x{C}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def depthwise_conv_3d_ndhwc_dhwcm():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    M = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)
    D = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)
    padding = 0

    K = choice_topped(KERNELS, (min(H, W, D) + 2 * padding - 1) // dilation - 1)

    H_ = ((H + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    W_ = ((W + 2 * padding - dilation * (K - 1) - 1) // stride) + 1
    D_ = ((D + 2 * padding - dilation * (K - 1) - 1) // stride) + 1

    return f"linalg.depthwise_conv_3d_ndhwc_dhwcm {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}x{C}x{M}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}x{M}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}x{M}xf32>"


def pooling_nchw_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nchw_max {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


def pooling_nchw_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nchw_sum {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


def pooling_ncw_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ncw_max {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def pooling_ncw_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ncw_sum {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{C}x{W}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{C}x{W_}xf32>) -> tensor<{N}x{C}x{W_}xf32>"


def pooling_ndhwc_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    D = choice(HEIGHTS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_max {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_ndhwc_min():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    D = choice(HEIGHTS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_min {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_ndhwc_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    D = choice(HEIGHTS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W, D) - 1) // dilation - 1)

    D_ = (D - dilation * (K - 1) - 1) // stride + 1
    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_ndhwc_sum {{dilations = dense<{dilation}> : tensor<3xi64>, strides = dense<{stride}> : tensor<3xi64>}} ins (%input, %filter: tensor<{N}x{D}x{H}x{W}x{C}xf32>, tensor<{K}x{K}x{K}xf32>) outs (%init: tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{D_}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_max {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_min():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_min {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nhwc_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    H = choice(HEIGHTS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (min(H, W) - 1) // dilation - 1)

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nhwc_sum {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{H}x{W}x{C}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{H_}x{W_}x{C}xf32>) -> tensor<{N}x{H_}x{W_}x{C}xf32>"


def pooling_nwc_max():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nwc_max {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"


def pooling_nwc_sum():
    N = choice(BATCH_SIZES)
    C = choice(CHANNELS)
    W = choice(HEIGHTS)

    dilation = choice(DILATIONS)
    stride = choice(STRIDES)

    K = choice_topped(KERNELS, (W - 1) // dilation - 1)

    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    return f"linalg.pooling_nwc_sum {{dilations = dense<{dilation}> : tensor<1xi64>, strides = dense<{stride}> : tensor<1xi64>}} ins (%input, %filter: tensor<{N}x{W}x{C}xf32>, tensor<{K}xf32>) outs (%init: tensor<{N}x{W_}x{C}xf32>) -> tensor<{N}x{W_}x{C}xf32>"




def get_nested_loops_data(loops):
    
    lines = loops.split('\n')

    loops_detailed = {}
    loops_detailed["nested_loops"] = []
    loops_detailed["op_count"] = {'+':0, '-':0, '*':0, '/':0, 'exp':0,}
    loops_detailed["load_data"] = []
    loops_detailed["store_data"] = []

    maps = {}
    args_of_loops = []
    args_of_map = {}

    for line in lines:
        
        if "affine_map" in line:
            map_name, map_function = line.strip().split(' = ')
            map_function = map_function.split(' -> ')[1][1:-2]
            maps[map_name] = map_function
            
        
        elif "affine.apply" in line:
            new_op, _, _, *map_name__args = line.strip().split(' ')
            map_name__args = ' '.join(map_name__args)
            s = map_name__args.index('(')
            map_name, args = map_name__args[:s], map_name__args[s+1:-1].split(', ')            
            mapping_string = copy(maps[map_name])
            for i in range(len(args)):
                mapping_string = mapping_string.replace(f'd{i}', args[i])
            # print(new_op, map_name, args, maps[map_name], mapping_string)
            args_of_map[new_op] = mapping_string
            
        elif "affine.for" in line:
            _, arg, _, lower, _, upper, _ = line.strip().split(' ')
            # print(arg, lower, upper)
            loops_detailed["nested_loops"].append((arg, int(lower), int(upper), 1))
            args_of_loops.append(arg)
            
        elif "affine.load" in line:
            # print(line.strip().split(' ')[:-2])
            new_op, _, _, *alloc = line.strip().split(' ')[:-2]
            alloc = ' '.join(alloc)
            args = alloc.split('[')[1][:-1].split(', ')

            for i in range(len(args)):
                if args[i] in args_of_map:
                    args[i] = args_of_map[args[i]]
                    
            loops_detailed["load_data"].append(args)
        
        elif "arith.addf" in line:loops_detailed["op_count"]['+'] += 1
        elif "arith.mulf" in line:loops_detailed["op_count"]['*'] += 1
        elif "arith.subf" in line:loops_detailed["op_count"]['-'] += 1
        elif "arith.divf" in line:loops_detailed["op_count"]['/'] += 1
        elif "math.exp" in line:loops_detailed["op_count"]['exp'] += 1

    return loops_detailed


def remove_duplicate_args(args, shapes):
    args_shapes = list(zip(args, shapes))
    seen = set()
    result = []
    for item in args_shapes:
        if item not in seen:
            seen.add(item)
            result.append(item)
            
    args = [x for (x, _) in result]
    shapes = [x for (_, x) in result]
    return args, shapes


def transform_wrapper(operation):

    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    args, shapes = [], []
    for field in fields:
        args_field, shapes_field = field.split(':')
        args   += args_field.split(',')
        shapes += shapes_field.split(',')

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

    args, shapes = remove_duplicate_args(args, shapes)
    
    # print(args, shapes)
    
    #############################################################
    # consts:
    dims = []
    unique_dims = set()
    for shape in shapes:
        if shape.startswith("tensor"):
            arg_dims = list(map(int, re.findall(r'\d+', shape[7:-5])))
            dims.append( arg_dims )
            unique_dims = unique_dims.union(arg_dims)
        else: # shape == "f32"
            dims.append( -1 )
            unique_dims = unique_dims.union([-1])

    unique_dims = sorted(list(unique_dims))

    # print(unique_dims)
    
    consts_snippet = ""
    for dim in unique_dims:
        if dim != -1:
            consts_snippet += f"  %c{dim} = arith.constant {dim} : index\n"

    #############################################################
    # allocations:

    allocations_snippet = ""

    for arg, shape, arg_dims in zip(args, shapes, dims):
        # print(arg, shape, arg_dims)
        if shape.startswith("tensor"):
            n = shape.count("x")
            temp_shape = "tensor<" + "?x"*n + shape[-4:] # f32> or i64> ir i32>
            alloc_params = ", ".join([f"%c{dim}" for dim in arg_dims])
            allocations_snippet += f"  {arg}_temp = bufferization.alloc_tensor({alloc_params}) : {temp_shape}\n"
            allocations_snippet += f"  {arg} = tensor.cast {arg}_temp : {temp_shape} to {shape}\n"
        else:
            # print(arg, shape, arg_dims)
            allocations_snippet += f"  {arg} = arith.constant 1.00000e+00 : f32\n"

    # print(allocations_snippet)

    #############################################################
    # function call:

    function_call_snippet = f"  %ret_arg = func.call @func_call({', '.join(args)}) : ({', '.join(shapes)}) -> ({shapes[-1]})"

    #############################################################
    # All code:

    code = ""
    code += 'module attributes {torch.debug_module_name = "Net"} {\n'
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printFlops(f64)\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n"
    code += "func.func private @printMemrefF32(tensor<*xf32>)\n"
    code += "\n"
    code += "\n"
    code +=f"func.func @matmul() -> {shapes[-1]}{{\n"
    code += "\n"
    code += "%val = arith.constant 2.00000e+00 : f32\n"
    code += "%zero = arith.constant 0.00000e+00 : f32\n"
    code += "\n"
    
    # code +=f"%out = bufferization.alloc_tensor() : tensor<{N}x{K}xf32>\n"
    # code +=f"%A = linalg.fill ins(%val : f32) outs(%out : tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32>\n"
    for arg, shape, arg_dims in zip(args, shapes, dims):
        if shape != 'f32':
            tmp_arg = f'%tmp_{arg[1:]}'
            code +=f"{tmp_arg} = bufferization.alloc_tensor() : {shape}\n"
            code +=f"{arg} = linalg.fill ins(%val : f32) outs({tmp_arg} : {shape}) -> {shape}\n"
        else:
            code +=f"{arg} = arith.constant 2.00000e+00 : f32\n"
    
    code += "\n"
    code += "%t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "\n"
    
    # code +=f"%D = linalg.matmul ins(%A, %B: tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%C: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>\n"
    code += f"%return_arg = {operation}"
    
    code += "\n"
    code += "%t = func.call @nanoTime() : () -> (i64)\n"
    code += "%delta = arith.subi %t, %t0 : i64\n"
    code += "%fp = arith.uitofp %delta : i64 to f64\n"
    code += "// func.call @printFlops(%fp) : (f64) -> ()\n"
    code += "func.call @printI64(%delta) : (i64) -> ()\n"
    code += "func.call @printNewline() : () -> ()\n"
    code += "\n"
    code +=f"return %return_arg : {shapes[-1]} \n"
    code += "}\n"
    code += "\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code +=f"    %outputmain = func.call @matmul() : () -> {shapes[-1]}\n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"
    code += "}\n"

    return code



def evaluate_code(code, timeout=20):
    # command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs create-deallocs function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops -scf-foreach-thread-lowering  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_2 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libomp.so"""
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"
    # tmp_file = "generated_mlir/bigger_input_nn.mlir"
    
    os.environ["OMP_NUM_THREADS"] = "8"
    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    out = os.popen(f"""{command_1} {tmp_file} | {command_2} /dev/stdin""").read()
    # out = os.popen(f"""{command_1} {tmp_file}""").read()
    
    if out:
        return int(out.strip().split('\n')[-1])
    else:
        return None

def evaluate_code_wrapper(code, return_list):
    res = evaluate_code(code)
    return_list.append(res)

def evaluate_code_with_timeout(code, timeout):
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=evaluate_code_wrapper, args=(code, return_list))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # The function is still running, terminate the process
        process.terminate()
        process.join()

        return None
    else:
        # The function completed within the timeout
        return return_list[0]





LINALG_OPERATION_GENERATORS = {
    # "add": [add, SMALL],
    # "sub": [sub, SMALL],
    # "max": [max, SMALL],
    # "mul": [mul, SMALL],
    # "abs": [abs, SMALL],
    # "ceil": [ceil, SMALL],
    # "copy": [copy_, MEDIUM],
    # "fill": [fill, BIG],
    # "transpose": [transpose, BIG],
    # "batch_matmul": [batch_matmul, MEDIUM],
    # "batch_matmul_transpose_a": [batch_matmul_transpose_a, MEDIUM],
    # "batch_matmul_transpose_b": [batch_matmul_transpose_b, MEDIUM],
    # "batch_reduce_matmul": [batch_reduce_matmul, MEDIUM],
    "matmul": [matmul, 10],
    # "matmul_transpose_a": [matmul_transpose_a, MEDIUM],
    # "matmul_transpose_b": [matmul_transpose_b, MEDIUM],
    # "conv_1d": [conv_1d, MEDIUM],
    # "conv_1d_ncw_fcw": [conv_1d_ncw_fcw, MEDIUM],
    # "conv_1d_nwc_wcf": [conv_1d_nwc_wcf, MEDIUM],
    # "conv_2d": [conv_2d, 2000],
    "conv_2d_nchw_fchw": [conv_2d_nchw_fchw, 10],
    # "conv_2d_ngchw_fgchw": [conv_2d_ngchw_fgchw, MEDIUM],
    # "conv_2d_nhwc_fhwc": [conv_2d_nhwc_fhwc, MEDIUM],
    # "conv_2d_nhwc_hwcf": [conv_2d_nhwc_hwcf, MEDIUM],
    # "conv_3d": [conv_3d, MEDIUM],
    # "conv_3d_ncdhw_fcdhw": [conv_3d_ncdhw_fcdhw, MEDIUM],
    # "depthwise_conv_1d_ncw_cw": [depthwise_conv_1d_ncw_cw, MEDIUM],
    # "depthwise_conv_1d_nwc_wc": [depthwise_conv_1d_nwc_wc, MEDIUM],
    # "depthwise_conv_1d_nwc_wcm": [depthwise_conv_1d_nwc_wcm, MEDIUM],
    # "depthwise_conv_2d_nchw_chw": [depthwise_conv_2d_nchw_chw, MEDIUM],
    # "depthwise_conv_2d_nhwc_hwc": [depthwise_conv_2d_nhwc_hwc, MEDIUM],
    # "depthwise_conv_2d_nhwc_hwcm": [depthwise_conv_2d_nhwc_hwcm, MEDIUM],
    # "depthwise_conv_3d_ncdhw_cdhw": [depthwise_conv_3d_ncdhw_cdhw, MEDIUM],
    # "depthwise_conv_3d_ndhwc_dhwc": [depthwise_conv_3d_ndhwc_dhwc, MEDIUM],
    # "depthwise_conv_3d_ndhwc_dhwcm": [depthwise_conv_3d_ndhwc_dhwcm, MEDIUM],
    # "pooling_nchw_max": [pooling_nchw_max, MEDIUM],
    # "pooling_nchw_sum": [pooling_nchw_sum, MEDIUM],
    # "pooling_ncw_max": [pooling_ncw_max, MEDIUM],
    # "pooling_ncw_sum": [pooling_ncw_sum, MEDIUM],
    # "pooling_ndhwc_max": [pooling_ndhwc_max, MEDIUM],
    # "pooling_ndhwc_min": [pooling_ndhwc_min, MEDIUM],
    # "pooling_ndhwc_sum": [pooling_ndhwc_sum, MEDIUM],
    # "pooling_nhwc_max": [pooling_nhwc_max, MEDIUM],
    # "pooling_nhwc_min": [pooling_nhwc_min, MEDIUM],
    # "pooling_nhwc_sum": [pooling_nhwc_sum, MEDIUM],
    # "pooling_nwc_max": [pooling_nwc_max, MEDIUM],
    # "pooling_nwc_sum": [pooling_nwc_sum, MEDIUM],
}

print( sum( amount for operation_name, (generator, amount) in LINALG_OPERATION_GENERATORS.items() ) )

all_operations = {}

for operation_name, (generator, amount) in tqdm(LINALG_OPERATION_GENERATORS.items(), desc="linalg operations"):

    for i in tqdm(range(amount), desc=operation_name):
        
        raw_operation = generator()
        
        # print(raw_operation)
                
        wrapped_operation = function_wrapper(raw_operation)  
        loops = lower_linalg_to_loops(wrapped_operation)            
        
        loops_data = get_nested_loops_data(loops)
        
        transform_wrapped_operation = transform_wrapper(raw_operation)
        
        # continue
        exec_time = evaluate_code_with_timeout(transform_wrapped_operation, 30)
        
        if exec_time:
            all_operations[f"{raw_operation}"] = {
                "operation": raw_operation,
                "wrapped_operation": wrapped_operation,
                "lowered_operation": loops,
                "transform_wrapped_operation": transform_wrapped_operation,
                "loops_data": loops_data,
                "execution_time":exec_time,
            }
        else:
            continue
        
        # print(raw_operation, end='\n\n\n')
        # print(wrapped_operation, end='\n\n\n')
        # print(loops, end='\n\n\n')
        # print(transform_wrapped_operation, end='\n\n\n')
        # print(loops_data, end='\n\n\n')

    with open(f"./generated_data/10_matmul_10_conv.json", "w") as file:
        json.dump(all_operations, file)