module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1482 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node21__model.layer-0.kernel", tf_saved_model.exported_names = [], type = tensor<2x4xf32>, value = dense<[[-0.366711378, 0.366582394, 0.264327288, -0.515119076], [-0.232851505, -0.044178009, -0.418082476, -0.119741678]]> : tensor<2x4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node22__model.layer-0.bias", tf_saved_model.exported_names = [], type = tensor<4xf32>, value = dense<0.000000e+00> : tensor<4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node29__model.layer-1.kernel", tf_saved_model.exported_names = [], type = tensor<4x8xf32>, value = dense<[[-0.518020332, -0.0076329708, -0.273415715, 0.446661174, -0.602498293, -0.121825993, 0.0328769088, -0.199197114], [-0.368024915, 0.63398689, -0.601587892, 3.564610e-01, 0.312985241, -0.690208554, -0.603075683, -0.648099839], [0.237009108, 0.482332766, 0.0121293664, -0.034699142, 0.0742671489, -0.331233412, -0.581755817, -0.152040184], [-0.084192872, -0.14643079, -0.460976422, -0.41481635, 0.611420094, -0.458459258, -0.137784958, -0.554757535]]> : tensor<4x8xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node30__model.layer-1.bias", tf_saved_model.exported_names = [], type = tensor<8xf32>, value = dense<0.000000e+00> : tensor<8xf32>} : () -> ()
  func.func @__inference_my_predict_1000(%arg0: tensor<1x2xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<2x4xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node21__model.layer-0.kernel"}, %arg2: tensor<!tf_type.resource<tensor<4xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node22__model.layer-0.bias"}, %arg3: tensor<!tf_type.resource<tensor<4x8xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node29__model.layer-1.kernel"}, %arg4: tensor<!tf_type.resource<tensor<8xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node30__model.layer-1.bias"}) -> (tensor<1x8xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful, tf_saved_model.exported_names = ["my_predict"]} {
    %0 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<!tf_type.resource<tensor<2x4xf32>>>) -> tensor<!tf_type.resource>
    %1 = "tf.Cast"(%arg2) {Truncate = false} : (tensor<!tf_type.resource<tensor<4xf32>>>) -> tensor<!tf_type.resource>
    %2 = "tf.Cast"(%arg3) {Truncate = false} : (tensor<!tf_type.resource<tensor<4x8xf32>>>) -> tensor<!tf_type.resource>
    %3 = "tf.Cast"(%arg4) {Truncate = false} : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<!tf_type.resource>
    %4 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%3) {device = ""} : (tensor<!tf_type.resource>) -> tensor<8xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4x8xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.ReadVariableOp"(%1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource>) -> tensor<2x4xf32>
      %control_6 = tf_executor.island(%control_5, %control_3, %control_1, %control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_7, %control_8 = tf_executor.island wraps "tf.MatMul"(%arg0, %outputs_4) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x2xf32>, tensor<2x4xf32>) -> tensor<1x4xf32>
      %outputs_9, %control_10 = tf_executor.island wraps "tf.BiasAdd"(%outputs_7, %outputs_2) {data_format = "NHWC", device = ""} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
      %outputs_11, %control_12 = tf_executor.island wraps "tf.Relu"(%outputs_9) {device = ""} : (tensor<1x4xf32>) -> tensor<1x4xf32>
      %outputs_13, %control_14 = tf_executor.island wraps "tf.MatMul"(%outputs_11, %outputs_0) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x4xf32>, tensor<4x8xf32>) -> tensor<1x8xf32>
      %outputs_15, %control_16 = tf_executor.island wraps "tf.BiasAdd"(%outputs_13, %outputs) {data_format = "NHWC", device = ""} : (tensor<1x8xf32>, tensor<8xf32>) -> tensor<1x8xf32>
      %outputs_17, %control_18 = tf_executor.island wraps "tf.Softmax"(%outputs_15) {device = ""} : (tensor<1x8xf32>) -> tensor<1x8xf32>
      %outputs_19, %control_20 = tf_executor.island(%control_6) wraps "tf.Identity"(%outputs_17) {device = ""} : (tensor<1x8xf32>) -> tensor<1x8xf32>
      tf_executor.fetch %outputs_19, %control_5, %control_3, %control_1, %control : tensor<1x8xf32>, !tf_executor.control, !tf_executor.control, !tf_executor.control, !tf_executor.control
    }
    return %4 : tensor<1x8xf32>
  }
  func.func private @__inference__wrapped_model_1500(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x8xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg4) {device = ""} : (tensor<!tf_type.resource>) -> tensor<8xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4x8xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<2x4xf32>
      %control_6 = tf_executor.island(%control_5, %control_3, %control_1, %control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_7, %control_8 = tf_executor.island wraps "tf.MatMul"(%arg0, %outputs_4) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x2xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>
      %outputs_9, %control_10 = tf_executor.island wraps "tf.BiasAdd"(%outputs_7, %outputs_2) {data_format = "NHWC", device = ""} : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
      %outputs_11, %control_12 = tf_executor.island wraps "tf.Relu"(%outputs_9) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
      %outputs_13, %control_14 = tf_executor.island wraps "tf.MatMul"(%outputs_11, %outputs_0) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x8xf32>) -> tensor<?x8xf32>
      %outputs_15, %control_16 = tf_executor.island wraps "tf.BiasAdd"(%outputs_13, %outputs) {data_format = "NHWC", device = ""} : (tensor<?x8xf32>, tensor<8xf32>) -> tensor<?x8xf32>
      %outputs_17, %control_18 = tf_executor.island wraps "tf.Softmax"(%outputs_15) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
      %outputs_19, %control_20 = tf_executor.island(%control_6) wraps "tf.Identity"(%outputs_17) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
      tf_executor.fetch %outputs_19, %control_5, %control_3, %control_1, %control : tensor<?x8xf32>, !tf_executor.control, !tf_executor.control, !tf_executor.control, !tf_executor.control
    }
    return %0 : tensor<?x8xf32>
  }
  func.func private @__inference_dense_1_layer_call_and_return_conditional_losses_1790(%arg0: tensor<?x4xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x8xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x4>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<8xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4x8xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.MatMul"(%arg0, %outputs_0) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x8xf32>) -> tensor<?x8xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.BiasAdd"(%outputs_2, %outputs) {data_format = "NHWC", device = ""} : (tensor<?x8xf32>, tensor<8xf32>) -> tensor<?x8xf32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Softmax"(%outputs_4) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
      %control_8 = tf_executor.island(%control_1, %control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_9, %control_10 = tf_executor.island(%control_8) wraps "tf.Identity"(%outputs_6) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
      tf_executor.fetch %outputs_9, %control_1, %control : tensor<?x8xf32>, !tf_executor.control, !tf_executor.control
    }
    return %0 : tensor<?x8xf32>
  }
  func.func private @__inference_dense_1_layer_call_and_return_conditional_losses_2840(%arg0: tensor<?x4xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x8xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x4>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<8xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4x8xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.MatMul"(%arg0, %outputs_0) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x8xf32>) -> tensor<?x8xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.BiasAdd"(%outputs_2, %outputs) {data_format = "NHWC", device = ""} : (tensor<?x8xf32>, tensor<8xf32>) -> tensor<?x8xf32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Softmax"(%outputs_4) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
      %control_8 = tf_executor.island(%control_1, %control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_9, %control_10 = tf_executor.island(%control_8) wraps "tf.Identity"(%outputs_6) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
      tf_executor.fetch %outputs_9, %control_1, %control : tensor<?x8xf32>, !tf_executor.control, !tf_executor.control
    }
    return %0 : tensor<?x8xf32>
  }
  func.func private @__inference_dense_1_layer_call_fn_2730(%arg0: tensor<?x4xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "267"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "269"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x4>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_1_layer_call_and_return_conditional_losses_1790} : (tensor<?x4xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
      %control_0 = tf_executor.island(%control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_1, %control_2 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outputs_1, %control : tensor<?x?xf32>, !tf_executor.control
    }
    return %0 : tensor<?x?xf32>
  }
  func.func private @__inference_dense_layer_call_and_return_conditional_losses_1630(%arg0: tensor<?x2xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x4xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<2x4xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.MatMul"(%arg0, %outputs_0) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x2xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.BiasAdd"(%outputs_2, %outputs) {data_format = "NHWC", device = ""} : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Relu"(%outputs_4) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
      %control_8 = tf_executor.island(%control_1, %control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_9, %control_10 = tf_executor.island(%control_8) wraps "tf.Identity"(%outputs_6) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
      tf_executor.fetch %outputs_9, %control_1, %control : tensor<?x4xf32>, !tf_executor.control, !tf_executor.control
    }
    return %0 : tensor<?x4xf32>
  }
  func.func private @__inference_dense_layer_call_and_return_conditional_losses_2640(%arg0: tensor<?x2xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x4xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<2x4xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.MatMul"(%arg0, %outputs_0) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x2xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.BiasAdd"(%outputs_2, %outputs) {data_format = "NHWC", device = ""} : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Relu"(%outputs_4) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
      %control_8 = tf_executor.island(%control_1, %control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_9, %control_10 = tf_executor.island(%control_8) wraps "tf.Identity"(%outputs_6) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
      tf_executor.fetch %outputs_9, %control_1, %control : tensor<?x4xf32>, !tf_executor.control, !tf_executor.control
    }
    return %0 : tensor<?x4xf32>
  }
  func.func private @__inference_dense_layer_call_fn_2530(%arg0: tensor<?x2xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "247"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "249"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_layer_call_and_return_conditional_losses_1630} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
      %control_0 = tf_executor.island(%control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_1, %control_2 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outputs_1, %control : tensor<?x?xf32>, !tf_executor.control
    }
    return %0 : tensor<?x?xf32>
  }
  func.func private @__inference_sequential_layer_call_and_return_conditional_losses_1860(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "164"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "166"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "180"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "182"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_layer_call_and_return_conditional_losses_1630} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.StatefulPartitionedCall"(%outputs, %arg3, %arg4) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_1_layer_call_and_return_conditional_losses_1790} : (tensor<?x?xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
      %control_2 = tf_executor.island(%control, %control_1) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_3, %control_4 = tf_executor.island(%control_2) wraps "tf.Identity"(%outputs_0) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outputs_3, %control, %control_1 : tensor<?x?xf32>, !tf_executor.control, !tf_executor.control
    }
    return %0 : tensor<?x?xf32>
  }
  func.func private @__inference_sequential_layer_call_and_return_conditional_losses_2000(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "189"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "191"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "194"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "196"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_layer_call_and_return_conditional_losses_1630} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.StatefulPartitionedCall"(%outputs, %arg3, %arg4) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_1_layer_call_and_return_conditional_losses_1790} : (tensor<?x?xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
      %control_2 = tf_executor.island(%control, %control_1) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_3, %control_4 = tf_executor.island(%control_2) wraps "tf.Identity"(%outputs_0) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outputs_3, %control, %control_1 : tensor<?x?xf32>, !tf_executor.control, !tf_executor.control
    }
    return %0 : tensor<?x?xf32>
  }
  func.func private @__inference_sequential_layer_call_fn_2130(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "203"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "205"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "207"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "209"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2, %arg3, %arg4) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2, 3, 4], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_sequential_layer_call_and_return_conditional_losses_1860} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
      %control_0 = tf_executor.island(%control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_1, %control_2 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outputs_1, %control : tensor<?x?xf32>, !tf_executor.control
    }
    return %0 : tensor<?x?xf32>
  }
  func.func private @__inference_sequential_layer_call_fn_2260(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "216"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "218"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "220"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "222"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2, %arg3, %arg4) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2, 3, 4], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_sequential_layer_call_and_return_conditional_losses_2000} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
      %control_0 = tf_executor.island(%control) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_1, %control_2 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outputs_1, %control : tensor<?x?xf32>, !tf_executor.control
    }
    return %0 : tensor<?x?xf32>
  }
}