��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
b
serving_default_xPlaceholder*
_output_shapes

:*
dtype0*
shape
:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_xdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference_signature_wrapper_114

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
/
	model

my_predict

signatures*
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature*

trace_0* 

serving_default* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
* 
�
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

$trace_0
%trace_1* 

&trace_0
'trace_1* 
* 
* 
* 

0
1*

0
1*
* 
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

-trace_0* 

.trace_0* 
b\
VARIABLE_VALUEdense/kernel<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
dense/bias:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

4trace_0* 

5trace_0* 
d^
VARIABLE_VALUEdense_1/kernel<model/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEdense_1/bias:model/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *%
f R
__inference__traced_save_330
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_restore_351��
�

�
@__inference_dense_1_layer_call_and_return_conditional_losses_179

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
>__inference_dense_layer_call_and_return_conditional_losses_163

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_200
dense_input
	dense_189:
	dense_191:
dense_1_194:
dense_1_196:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input	dense_189	dense_191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_163�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_194dense_1_196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_179w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:#

_user_specified_name196:#

_user_specified_name194:#

_user_specified_name191:#

_user_specified_name189:T P
'
_output_shapes
:���������
%
_user_specified_namedense_input
�	
�
(__inference_sequential_layer_call_fn_213
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_186o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:#

_user_specified_name209:#

_user_specified_name207:#

_user_specified_name205:#

_user_specified_name203:T P
'
_output_shapes
:���������
%
_user_specified_namedense_input
�
�
#__inference_dense_layer_call_fn_253

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:#

_user_specified_name249:#

_user_specified_name247:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_dense_1_layer_call_fn_273

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_179o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:#

_user_specified_name269:#

_user_specified_name267:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference__wrapped_model_150
dense_inputA
/sequential_dense_matmul_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:
identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:����������
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������s
IdentityIdentity$sequential/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:T P
'
_output_shapes
:���������
%
_user_specified_namedense_input
�

�
>__inference_dense_layer_call_and_return_conditional_losses_264

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_my_predict_100
xA
/sequential_dense_matmul_readvariableop_resource:>
0sequential_dense_biasadd_readvariableop_resource:C
1sequential_dense_1_matmul_readvariableop_resource:@
2sequential_dense_1_biasadd_readvariableop_resource:
identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
sequential/dense/MatMulMatMulx.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes

:�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:s
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*
_output_shapes

:j
IdentityIdentity$sequential/dense_1/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:A =

_output_shapes

:

_user_specified_namex
�-
�
__inference__traced_save_330
file_prefix5
#read_disablecopyonread_dense_kernel:1
#read_1_disablecopyonread_dense_bias:9
'read_2_disablecopyonread_dense_1_kernel:3
%read_3_disablecopyonread_dense_1_bias:
savev2_const

identity_9��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHw
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes	
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_8Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_9IdentityIdentity_8:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
(__inference_sequential_layer_call_fn_226
dense_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_200o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:#

_user_specified_name222:#

_user_specified_name220:#

_user_specified_name218:#

_user_specified_name216:T P
'
_output_shapes
:���������
%
_user_specified_namedense_input
�
�
__inference__traced_restore_351
file_prefix/
assignvariableop_dense_kernel:+
assignvariableop_1_dense_bias:3
!assignvariableop_2_dense_1_kernel:-
assignvariableop_3_dense_1_bias:

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHz
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: v
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
_output_shapes
 "!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32$
AssignVariableOpAssignVariableOp:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
!__inference_signature_wrapper_114
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *#
fR
__inference_my_predict_100f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:#

_user_specified_name110:#

_user_specified_name108:#

_user_specified_name106:#

_user_specified_name104:A =

_output_shapes

:

_user_specified_namex
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_186
dense_input
	dense_164:
	dense_166:
dense_1_180:
dense_1_182:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input	dense_164	dense_166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_163�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_180dense_1_182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_179w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:#

_user_specified_name182:#

_user_specified_name180:#

_user_specified_name166:#

_user_specified_name164:T P
'
_output_shapes
:���������
%
_user_specified_namedense_input
�

�
@__inference_dense_1_layer_call_and_return_conditional_losses_284

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_defaulty
&
x!
serving_default_x:03
output_0'
StatefulPartitionedCall:0tensorflow/serving/predict:�E
I
	model

my_predict

signatures"
_generic_user_object
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_sequential
�
trace_02�
__inference_my_predict_100�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
�ztrace_0
,
serving_default"
signature_map
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

 layers
!metrics
"layer_regularization_losses
#layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
$trace_0
%trace_12�
(__inference_sequential_layer_call_fn_213
(__inference_sequential_layer_call_fn_226�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z$trace_0z%trace_1
�
&trace_0
'trace_12�
C__inference_sequential_layer_call_and_return_conditional_losses_186
C__inference_sequential_layer_call_and_return_conditional_losses_200�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z&trace_0z'trace_1
�B�
__inference__wrapped_model_150dense_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_my_predict_100x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
�
�B�
!__inference_signature_wrapper_114x"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
-trace_02�
#__inference_dense_layer_call_fn_253�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-trace_0
�
.trace_02�
>__inference_dense_layer_call_and_return_conditional_losses_264�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z.trace_0
:2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
4trace_02�
%__inference_dense_1_layer_call_fn_273�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z4trace_0
�
5trace_02�
@__inference_dense_1_layer_call_and_return_conditional_losses_284�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z5trace_0
 :2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_sequential_layer_call_fn_213dense_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_sequential_layer_call_fn_226dense_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_sequential_layer_call_and_return_conditional_losses_186dense_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_sequential_layer_call_and_return_conditional_losses_200dense_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_dense_layer_call_fn_253inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_dense_layer_call_and_return_conditional_losses_264inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_1_layer_call_fn_273inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_1_layer_call_and_return_conditional_losses_284inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_150o4�1
*�'
%�"
dense_input���������
� "1�.
,
dense_1!�
dense_1����������
@__inference_dense_1_layer_call_and_return_conditional_losses_284c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
%__inference_dense_1_layer_call_fn_273X/�,
%�"
 �
inputs���������
� "!�
unknown����������
>__inference_dense_layer_call_and_return_conditional_losses_264c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� 
#__inference_dense_layer_call_fn_253X/�,
%�"
 �
inputs���������
� "!�
unknown���������a
__inference_my_predict_100C!�
�
�
x
� "�
unknown�
C__inference_sequential_layer_call_and_return_conditional_losses_186r<�9
2�/
%�"
dense_input���������
p

 
� ",�)
"�
tensor_0���������
� �
C__inference_sequential_layer_call_and_return_conditional_losses_200r<�9
2�/
%�"
dense_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
(__inference_sequential_layer_call_fn_213g<�9
2�/
%�"
dense_input���������
p

 
� "!�
unknown����������
(__inference_sequential_layer_call_fn_226g<�9
2�/
%�"
dense_input���������
p 

 
� "!�
unknown���������
!__inference_signature_wrapper_114Z&�#
� 
�

x�
x"*�'
%
output_0�
output_0