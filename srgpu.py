import time
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from collections import defaultdict, namedtuple
from pathlib import Path
import tensorrt as trt
import cv2
import torch.nn.functional as F  # 导入 interpolate


# import pycuda.driver as cuda
# import pycuda.autoinit


class SuperResolutionEngine:
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, engine_path: str):
        self.weight = Path(engine_path) if isinstance(engine_path, str) else engine_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stream = torch.cuda.Stream(device=self.device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()
        num_bindings = model.num_bindings
        names = [model.get_binding_name(i) for i in range(num_bindings)]

        self.bindings: List[int] = [0] * num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_bindings = num_bindings
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.idx = list(range(self.num_outputs))

    def __init_bindings(self) -> None:
        idynamic = odynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))  # 创建一个元组类
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]  # 类型转换匹配
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                idynamic |= True
            inp_info.append(Tensor(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                odynamic |= True
            out_info.append(Tensor(name, dtype, shape))

        if not odynamic:
            self.output_tensor = [
                torch.empty(info.shape, dtype=info.dtype, device=self.device)
                for info in out_info
            ]
        self.idynamic = idynamic
        self.odynamic = odynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def infer(self, *inputs) -> Union[Tuple, torch.Tensor]:

        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[torch.Tensor] = [
            i.contiguous() for i in inputs
        ]

        for i in range(self.num_inputs):
            self.bindings[i] = contiguous_inputs[i].data_ptr()
            if self.idynamic:
                self.context.set_binding_shape(
                    i, tuple(contiguous_inputs[i].shape))

        outputs: List[torch.Tensor] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.odynamic:
                shape = tuple(self.context.get_binding_shape(j))
                output = torch.empty(size=shape,
                                     dtype=self.out_info[i].dtype,
                                     device=self.device)
            else:
                output = self.output_tensor[i]
            self.bindings[j] = output.data_ptr()
            outputs.append(output)

        self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
        self.stream.synchronize()

        return tuple(outputs[i]
                     for i in self.idx) if len(outputs) > 1 else outputs[0]

    def get4srimg(self, img: np.ndarray, sizes: tuple) -> np.ndarray:
        # start_time = time.time()
        lr_image = (img / 255).astype(np.float32)
        lr_image = np.transpose(lr_image, (2, 0, 1))
        lr_image = self.ndarray2tensor(lr_image)
        lr_image = torch.unsqueeze(lr_image, 0)

        sr_image = self.infer(lr_image)
        reversed_tuple = sizes[::-1]
        # print(sr_image.shape)
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if cuda_available:
            # print("CUDA is available, using GPU for resize.")
            gpu_resize_flag = True
            sr_image_gpu_torch_resized = F.interpolate(
                sr_image,  # [1, 3, H_in, W_in]
                size=reversed_tuple,  # (H_out, W_out)
                mode='area')
        else:
            gpu_resize_flag = False
            # print("CUDA not available, using CPU for resize.")

        # print(sr_image.shape)
        image = self.process_output(sr_image_gpu_torch_resized)
        # temp = time.time()
        # print(f"超分推理 {(temp - start_time) * 1000:.2f} ms")
        return image, gpu_resize_flag

    def ndarray2tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.tensor(img, device=self.device)

    def process_output(self, sr_image: np.ndarray) -> np.ndarray:
        # sr_image=torch.from_numpy(sr_image)
        sr_image = sr_image * 255
        sr_image = sr_image.clamp(0, 255)  # tensor的属性
        sr_image = sr_image.cpu().numpy()
        sr_image = sr_image[0]
        sr_image = np.transpose(sr_image, (1, 2, 0)).round().astype(np.uint8)
        return sr_image


# 使用示例
if __name__ == "__main__":
    checkpoint = "m10c32c3.engine"
    sr_engine = SuperResolutionEngine(checkpoint)
    cap = cv2.VideoCapture("2562.mp4")
    if not cap.isOpened():
        print("Error: Could not open video device.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        height, width, channel = frame.shape

        resize_frame = cv2.resize(frame, (640, 512), interpolation=cv2.INTER_AREA)
        cv2.imshow('frame', resize_frame)
        start_time = time.time()
        sr_image, flag = sr_engine.get4srimg(frame, (640, 512))
        temp = time.time()
        print(f"超分推理 {(temp - start_time) * 1000:.2f} ms")
        # sr_image = cv2.resize(sr_image, (640, 512), interpolation=cv2.INTER_AREA)
        cv2.imshow('resize Super-Resolution', sr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

