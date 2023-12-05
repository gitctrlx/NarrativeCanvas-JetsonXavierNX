import os
import tensorrt as trt

from models.Calibrator import EntropyCalibrator2, MinMaxCalibrator 

class Builder:
    def __init__(self, onnxFile, trtFile, accuracy,
                 nBatchSize=[1, 4, 8], nChannel=[3, 3, 3], nHeight=[224, 224, 224], nWidth=[224, 224, 224], 
                 calibrator='EntropyCalibrator2', calibrationDataPath='./calibdata/', nCalibration=120, int8cacheFile='./engine/int8Cache/int8.cache', 
                 bUseTimeCache=True, timingCacheFile = "./engine/timingCache/model.TimingCache", bIgnoreMismatch = False,
                 removePlanCache=False):
        
        self.onnxFile = onnxFile
        self.trtFile = trtFile
        self.accuracy = accuracy
        self.nBatchSize = nBatchSize
        self.nChannel = nChannel
        self.nHeight = nHeight
        self.nWidth = nWidth
        self.calibrator = calibrator
        self.calibrationDataPath = calibrationDataPath
        self.nCalibration = nCalibration
        self.int8cacheFile = int8cacheFile
        self.bUseTimeCache = bUseTimeCache
        self.timingCacheFile = timingCacheFile
        self.bIgnoreMismatch = bIgnoreMismatch # turn on if we allow the timing cache file using among different device
        self.removePlanCache = removePlanCache
        self.logger = trt.Logger(trt.Logger.INFO)

    def build_model(self):
        print("[INFO] Model building start.")
        if self.removePlanCache:
            os.system(f"rm -rf {self.trtFile} {self.int8cacheFile} {self.timingCacheFile}")

        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        timingCacheString = b""
        if self.bUseTimeCache and os.path.isfile(self.timingCacheFile):
            with open(self.timingCacheFile, "rb") as f:
                timingCacheString = f.read()
            if timingCacheString == None:
                print("[INFO]Failed loading %s" % self.timingCacheFile)
                return
            print("[INFO]Succeeded loading %s" % self.timingCacheFile)

        # Set timing cache flag
        if self.bUseTimeCache:
            timingCache = config.create_timing_cache(timingCacheString)
            #timingCache.reset()  # clean the timing cache, not required
            config.set_timing_cache(timingCache, self.bIgnoreMismatch)

        # Set precision flags based on accuracy
        if   self.accuracy == 'fp32':
            pass
        elif self.accuracy == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.accuracy == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            if self.calibrator == 'EntropyCalibrator2':
                config.int8_calibrator = EntropyCalibrator2(self.calibrationDataPath, self.nCalibration, 
                                                            (self.nBatchSize[1], self.nChannel[1], self.nHeight[1], self.nWidth[1]), 
                                                            self.int8cacheFile)
            elif self.calibrator == 'MinMaxCalibrator':
                config.int8_calibrator = MinMaxCalibrator(self.calibrationDataPath, self.nCalibration, 
                                                            (self.nBatchSize[1], self.nChannel[1], self.nHeight[1], self.nWidth[1]), 
                                                            self.int8cacheFile)

        # Parse the ONNX file
        parser = trt.OnnxParser(network, self.logger)
        if not os.path.exists(self.onnxFile):
            print("[ERROR] Failed finding ONNX file!")
            return False

        with open(self.onnxFile, "rb") as model:
            if not parser.parse(model.read()):
                print("[ERROR] Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False

        # Set optimization profiles for different batch sizes
        profile = builder.create_optimization_profile()
        inputTensor = network.get_input(0)
        profile.set_shape(inputTensor.name, 
                          [self.nBatchSize[0], self.nChannel[0], self.nHeight[0], self.nWidth[0]],
                          [self.nBatchSize[1], self.nChannel[1], self.nHeight[1], self.nWidth[1]],
                          [self.nBatchSize[2], self.nChannel[2], self.nHeight[2], self.nWidth[2]])
        config.add_optimization_profile(profile)
        if self.accuracy == 'int8':
            config.set_calibration_profile(profile)

        # Build the engine
        engineString = builder.build_serialized_network(network, config)
        if engineString is None:
            print("[ERROR] Failed building engine!")
            return False

        # Save the engine to a file
        with open(self.trtFile, "wb") as f:
            f.write(engineString)

        # Save the timingcache to a file
        if self.bUseTimeCache:
            timingCacheNew = config.get_timing_cache()
            #res = timingCache.combine(timingCacheNew, bIgnoreMismatch)  # merge timing cache from the old one (load form file) with the new one (created by this build), not required
            timingCache = timingCacheNew
            #print("timingCache.combine:%s" % res)
            timeCacheString = timingCache.serialize()
            with open(self.timingCacheFile, "wb") as f:
                f.write(timeCacheString)
                print(f"Succeeded saving {self.timingCacheFile}")

        print("[PASSED] Model building successful.")
        return True
    


# if __name__ == '__main__':
#     model = 'resnet50_8xb256-rsb-a1-600e_in1k'
#     accuracy = 'int8'
#     pss = 'pss'
#     builder = Builder(onnxFile = f'./onnx/{model}-{pss}.onnx',  trtFile=f'./engine/{model}_{accuracy}.plan', 
#                       accuracy=accuracy,
#                       calibrationDataPath='./calibdata/',
#                       int8cacheFile = f'./engine/int8Cache/{model}.cache',
#                       timingCacheFile = f'./engine/timingCache/{model}.TimingCache',
#                       removePlanCache=False)
    
#     if builder.build_model():
#         # Assuming ImageClassifier is a defined class elsewhere in your code
#         from Infer import ImageClassifier
#         image_paths = ['./demo/demo.JPEG', './demo/demo2.JPEG', './demo/demo3.JPEG', './demo/demo4.JPEG']
#         predictions = ImageClassifier(trt_file = f'./engine/{model}_{accuracy}.plan', labels_file='./imagenet_classes.txt').predict(image_paths)
#         print(predictions)
#     else:
#         print("[ERROR] Model building failed.")

import argparse
if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Build a TensorRT engine from an ONNX model.')

    # Define required arguments
    parser.add_argument('--onnxFile', type=str, required=True, help='Path to the ONNX file.')
    parser.add_argument('--trtFile', type=str, required=True, help='Path to the TensorRT file.')
    parser.add_argument('--accuracy', type=str, choices=['fp32', 'fp16', 'int8'], default='int8', help='Precision accuracy.')
    parser.add_argument('--calibrationDataPath', type=str, default='./calibdata/', help='Path to calibration data.')
    parser.add_argument('--int8cacheFile', type=str, required=True, help='Path to the INT8 cache file.')
    parser.add_argument('--timingCacheFile', type=str, required=True, help='Path to the timing cache file.')
    parser.add_argument('--removePlanCache', action='store_true', help='Flag to remove plan cache.')

    # Parse the arguments
    args = parser.parse_args()

    # Create a Builder instance with the parsed arguments
    builder = Builder(onnxFile=args.onnxFile, trtFile=args.trtFile, 
                      accuracy=args.accuracy,
                      calibrationDataPath=args.calibrationDataPath,
                      int8cacheFile=args.int8cacheFile,
                      timingCacheFile=args.timingCacheFile,
                      removePlanCache=args.removePlanCache)
    
    # Build the model
    if builder.build_model():
        print("[INFO] Successful.")
    else:
        print("[ERROR] Model building failed.")

'''
python3 ./models/Build.py --onnxFile './models/onnx/resnet50_8xb256-rsb-a1-600e_in1k-pss.onnx' \
                    --trtFile './models/engine/example_fp16.plan' \
                    --accuracy fp16 \
                    --calibrationDataPath './models/calibdata/' \
                    --int8cacheFile './models/engine/int8Cache/example.cache' \
                    --timingCacheFile './models/engine/timingCache/example.TimingCache' \
                    --removePlanCache
'''