import time
import cv2
import random
import numpy as np
from numpy import ndarray
from typing import List, Tuple, Union
from torch import Tensor
import os
import pickle
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Union


import tensorrt as trt
import torch

class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
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
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
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

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def set_desired(self, desired: Optional[Union[List, Tuple]]):
        if isinstance(desired,
                      (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> Union[Tuple, torch.Tensor]:

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

def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[0] - new_unpad[0]) / 2, (new_shape[1] - new_unpad[1]) / 2

    # Resize if necessary
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    seg = im.astype(np.float32) / 255 if return_seg else None
    im = im.transpose([2, 0, 1])[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    return (im, seg) if return_seg else im


def det_postprocess(data: Tuple[Tensor, Tensor, Tensor, Tensor]):
    num_dets, bboxes, scores, labels = data
    num_dets, bboxes, scores, labels = num_dets[0], bboxes[0], scores[0], labels[0]
    nums = num_dets.item()
    if nums == 0:
        return bboxes.new_zeros((0, 4)), scores.new_zeros((0,)), labels.new_zeros((0,))
    scores[scores < 0] += 1
    return bboxes[:nums], scores[:nums], labels[:nums]


class YOLOv8Inference:
    def __init__(self, engine_file: str, device: torch.device):
        
        self.device = device
        self.engine = TRTModule(engine_file, self.device)
        self.H, self.W = self.engine.inp_info[0].shape[-2:]
        self.engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        self.CLASSES_DET = ("person", "bike", "car", "motor", "bus", "truck", "dog", "cyclist")
        self.COLORS = {cls: [random.randint(0, 255) for _ in range(3)] for cls in self.CLASSES_DET}

    def preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, float, float]:
        bgr, ratio, dwdh = letterbox(img, (self.W, self.H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb)
        dwdh = torch.as_tensor(dwdh * 2, dtype=torch.float32, device=self.device)
        tensor = torch.as_tensor(tensor, device=self.device)
        return tensor, ratio, dwdh

    def postprocess(self, bboxes: Tensor, scores: Tensor, labels: Tensor, dwdh: Tensor, ratio: float):
        bboxes = (bboxes - dwdh) / ratio
        return bboxes, scores, labels

    def infer(self, img: np.ndarray, resizes=(800,600)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w,h= img.shape[1], img.shape[0]
        tensor, ratio, dwdh = self.preprocess(img)
        data = self.engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        bboxes, scores, labels = self.postprocess(bboxes, scores, labels, dwdh, ratio)
        ratio= resizes[1] / h
        ratio2= resizes[0] /w
        # return bboxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
        if ratio2 == ratio:
            bboxes = bboxes*ratio
            torch.cuda.empty_cache()
            return bboxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
        else:
            print("推理图片不是等比例缩放")
            

# 摄像头推理测试
if __name__ == "__main__":
    engine_file = "edge0410_256.engine"
    device = torch.device('cuda:0')
    yolo_inference = YOLOv8Inference(engine_file, device)

    cap = cv2.VideoCapture("2562.mp4")
    if not cap.isOpened():
        print("Error: Could not open video device.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        time1=time.perf_counter()*1000
        bboxes, scores, labels = yolo_inference.infer(frame, (800, 600))
        time2=time.perf_counter()*1000-time1
        print(f"推理时间: {time2:.2f}秒")
        rzsize_frame = cv2.resize(frame, (800, 600))
       
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(rzsize_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rzsize_frame, f'Label: {label} Score: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow('YOLOv8 Inference', rzsize_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
