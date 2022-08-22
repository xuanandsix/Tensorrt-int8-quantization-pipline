
# import pdb;pdb.set_trace()
import sys
sys.path.append('trt')
import common
import tensorrt as trt
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit


# The Onnx path is used for Onnx models.
def build_engine_onnx(TRT_LOGGER, model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        # import pdb;pdb.set_trace()
        # builder.int8_mode = True
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)

def build_engine_onnx_int8(TRT_LOGGER, model_file, calib):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        # import pdb;pdb.set_trace()
        builder.int8_mode = True
        builder.int8_calibrator = calib

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        # import pdb;pdb.set_trace()
        return builder.build_cuda_engine(network)

def build_engine(TRT_LOGGER, model_file):
    
    with open(model_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())        

    return engine


