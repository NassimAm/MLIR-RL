# pylint: skip-file

from utils.observation_utils import AutoScheduleOperation
from random import randint, choice, shuffle
from tqdm import tqdm
import json

BATCH_SIZES = [4, 8, 16, 32, 64, 256, 512]
# HEIGHTS = [2**power for power in range(5, 11)]
HEIGHTS = [2**power for power in range(8, 13)]
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


def copy():
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
    N = choice(HEIGHTS)
    K = choice(HEIGHTS)
    M = choice(HEIGHTS)
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





LINALG_OPERATION_GENERATORS = {
    # "add": [add, SMALL],
    # "sub": [sub, SMALL],
    # "max": [max, SMALL],
    # "mul": [mul, SMALL],
    # "abs": [abs, SMALL],
    # "ceil": [ceil, SMALL],
    # "copy": [copy, MEDIUM],
    # "fill": [fill, BIG],
    # "transpose": [transpose, BIG],
    # "batch_matmul": [batch_matmul, MEDIUM],
    # "batch_matmul_transpose_a": [batch_matmul_transpose_a, MEDIUM],
    # "batch_matmul_transpose_b": [batch_matmul_transpose_b, MEDIUM],
    # "batch_reduce_matmul": [batch_reduce_matmul, MEDIUM],
    "matmul": [matmul, 3000],
    # "matmul_transpose_a": [matmul_transpose_a, MEDIUM],
    # "matmul_transpose_b": [matmul_transpose_b, MEDIUM],
    # "conv_1d": [conv_1d, MEDIUM],
    # "conv_1d_ncw_fcw": [conv_1d_ncw_fcw, MEDIUM],
    # "conv_1d_nwc_wcf": [conv_1d_nwc_wcf, MEDIUM],
    # "conv_2d": [conv_2d, 2000],
    # "conv_2d_nchw_fchw": [conv_2d_nchw_fchw, MEDIUM],
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
        raw_operation = "linalg.matmul ins(%arg0, %arg1 : tensor<1200x1500xf32>, tensor<1500x1000xf32>) outs(%arg2 : tensor<1200x1000xf32>) -> tensor<1200x1000xf32>"
        operation = AutoScheduleOperation(raw_operation)

        all_operations[f"{raw_operation}_{i}"] = {
            "operation": operation.operation,
            "wrapped_operation": operation.wrapped_operation,
            "lowered_operation": operation.lowered_operation,
            "transform_wrapped_operation": operation.transform_wrapped_operation,
            "loops_data": operation.loops_data,
        }

        break
    # break

    with open(f"./generated_data/linalg_one_matmul_generated_operations.json", "w") as file:
        json.dump(all_operations, file)