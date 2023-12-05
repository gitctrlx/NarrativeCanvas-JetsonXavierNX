import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

class ImageClassifier:
    
    def __init__(self, trt_file, labels_file='imagenet_classes.txt', nHeight=224, nWidth=224):
        self.trt_file = trt_file
        self.labels_file = labels_file
        self.class_names = self._read_class_names(self.labels_file)
        self.nHeight = nHeight
        self.nWidth = nWidth

        # Initialize CUDA device and context
        cuda.init()  # Initialize CUDA
        self.cuda_device = cuda.Device(0)  # Assuming using the first device
        self.cuda_context = self.cuda_device.make_context()

        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine = self._load_engine(self.trt_file)
        self._initialize_bindings() # Initialize input and output binding
        self.stream = cuda.Stream() # Create CUDA stream

    def __del__(self):
        # Clean up resources
        self.stream = None
        if self.cuda_context:
            self.cuda_context.pop()
            del self.cuda_context
            self.cuda_context = None

    def _initialize_bindings(self):
        self.num_bindings = self.engine.num_bindings
        self.input_bindings = []
        self.output_bindings = []

        for i in range(self.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_bindings.append(i)
            else:
                self.output_bindings.append(i)

    def _read_class_names(self, file_path: str):
        try:
            with open(file_path, 'r') as f:
                return [line.strip().split(', ')[1] for line in f.readlines()]
        except FileNotFoundError:
            raise Exception(f"[ERROR]Class labels file '{file_path}' not found.")

    def _load_engine(self, trtFile: str) -> trt.ICudaEngine:
        try:
            with open(trtFile, "rb") as f:
                engine_data = f.read()
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(engine_data)
        except FileNotFoundError:
            raise Exception(f"[ERROR]TensorRT engine file '{trtFile}' not found.")

    def _preprocess_image(self, image_path):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"[ERROR]The image {image_path} could not be found.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.nWidth, self.nHeight), interpolation=cv2.INTER_CUBIC)
        img_resized = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_resized - mean) / std
        return img_normalized.transpose(2, 0, 1)

    def _run_inference(self, data):
        # Execute inference in the current CUDA context
        context = self.engine.create_execution_context()

        # Set dynamic input shape
        batch_size = data.shape[0]
        context.set_binding_shape(0, (batch_size, 3, self.nHeight, self.nWidth))

        # Allocate host and device memory
        bufferH = [None] * self.num_bindings
        bufferD = [None] * self.num_bindings

        # Allocate memory for input data
        for i in self.input_bindings:
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            bufferH[i] = np.ascontiguousarray(data, dtype=dtype)
            bufferD[i] = cuda.mem_alloc(bufferH[i].nbytes)

        # Allocate memory for output data
        output_shape = list(self.engine.get_binding_shape(self.output_bindings[0]))
        output_shape[0] = batch_size
        for i in self.output_bindings:
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            bufferH[i] = np.empty(output_shape, dtype=dtype)
            bufferD[i] = cuda.mem_alloc(bufferH[i].nbytes)

        # h2d
        for i in self.input_bindings:
            cuda.memcpy_htod_async(bufferD[i], bufferH[i], self.stream)

        # execute
        bindings = [int(b) for b in bufferD]
        context.execute_async_v2(bindings, self.stream.handle)

        # d2h
        for i in self.output_bindings:
            cuda.memcpy_dtoh_async(bufferH[i], bufferD[i], self.stream)

        # Synchronize CUDA flow to ensure all operations are completed
        self.stream.synchronize()

        # Return the first output result
        return bufferH[self.output_bindings[0]]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, image_paths):
        batch_data = np.stack([self._preprocess_image(img_path) for img_path in image_paths])
        output = self._run_inference(batch_data)
        predictions = []
        for i in range(output.shape[0]):
            probabilities = self.softmax(output[i])
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            label_probs = {self.class_names[j]: float(probabilities[j]) for j in top5_indices}
            predictions.append(label_probs)
        return predictions

# Usage
if __name__ == '__main__':
    accuracy='fp16'
    model = 'resnet50_8xb256-rsb-a1-600e_in1k'
    image_paths = ['./demo/demo.JPEG','./demo/demo2.JPEG','./demo/demo3.JPEG', './demo/demo4.JPEG','./demo/demo5.jpg','./demo/demo6.png','./demo/demo7.JPEG', './demo/demo8.JPEG']
    predictions = ImageClassifier(trt_file=f'./engine/{model}_{accuracy}.plan', labels_file='./imagenet_classes.txt').predict(image_paths)
    print(predictions)
