# YOLO Weights and tutorial
## In this tutorial I will show in detail how to deploy **YOLO**, and how to change **TDL SDK** configuration files
### In this github repository you will find files with Yolov8 model weights look carefully at the file name. Model(yolov8s, yolov8n etc.)_image size(640, 320, 224 etc.)_and applicable parameters (base or modified parameters (augmentation, hyperparameters and so on). I will also include ultralytics files in this github repository for future. In this ultralytics folder you can familiarize yourself with the metrics of each model. ultralytics - runs - detect - (further you will find a folder with the model you are specifically interested in), by going to the folder with the model you will see the results of the metrics.

# How to convert the model to ONNX format
- Dowload the YOLO.pt weights from this github
- Go to your IDE and open ultralytics folder with already imported model weights file in .pt format (choose your environment) go to the terminal and install ultralytics - pip install     ultralytics. After installing ultralytics, go to the ultralytics folder and create there main.py file, write this code to export the model to onnx format:
  
    ```python  
        from ultralytics import YOLO
        import types

        input_size = (640, 640)

        def forward2(self, x):
            x_reg = [self.cv2[i](x[i]) for i in range(self.nl)]
            x_cls = [self.cv3[i](x[i]) for i in range(self.nl)]
            return x_reg + x_cls

        model_path = "./path_to_the_weights_of_yourmodel.pt"
        model = YOLO(model_path)
        model.model.model[-1].forward = types.MethodType(forward2, model.model.model[-1])
        model.export(format='onnx', opset=11, imgsz=input_size)
    ``` 

# TDL SDK ON YOUR LOCAL MACHINE (local terminal)
- Once you have the onnx model file, let's install the TDL SDK on your local machine. Open the terminal on your local machine
 1. Write there: wget https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/03/07/16/host-tools.tar.gz
 2. tar xvf host-tools.tar.gz
 3. cd host-tools
    export PATH=$PATH:$(pwd)/gcc/riscv64-linux-musl-x86_64/bin
 4. riscv64-unknown-linux-musl-gcc -v
 5. If everything's okay, you'll get this message: 
 
 6. Then download TDL SDK for DUO 256
 7. git clone https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x.git

# TPU MLIR ON YOUR LOCAL MACHINE (local terminal)
 1. git clone https://github.com/milkv-duo/tpu-mlir.git

# SETTING UP DOCKER ENVIRONMENT (local terminal)
 1. docker pull sophgo/tpuc_dev:v3.1
 2. docker run --privileged --name <name_of_your_container> -v $(pwd):/workspace -p 3000:3000 -it sophgo/tpuc_dev:v3.1 (it will creates the folder on your local machine with the all docker files)

# Copy all downloaded files and model weights in onnx format to the docker environment
 - Copy cvitek_tdl_sdk folder
  - docker cp <path>/home/max/cvitek-tdl-sdk-sg200x <container_name>:/workspace
 - Copy HOST-TOOLS folder 
  - docker cp <path>/home/max/host-tools <container_name>:/workspace
 - Copy ZIP Host-tools.tar.gz 
  - docker cp <path>/home/max/host-tools.tar.gz <container_name>:/workspace
 - Copy TPU-MLIR folder
  - docker cp <path>/home/max/tpu-mlir <container_name>:/workspace
 - Also you can copy 100 test images (we will use them while converting the model into .cvimodel format). *you can find 100 test images in github (test folder)
  - docker cp <path>/home/max/test <container_name>:/workspace
 - And copy one test image (also you can find in github test.jpg)
  - docker cp <path>/home/max/test.jpg <container_name>:/workspace
 - Copy the model weights in ONNX format 
  - docker cp <path>/home/max/best.onnx <container_name>:/workspace

# Then go to the docker environment
 1. (Local machine terminal) docker start <name_of_the_container>
 2. (Local machine terminal) Docker attach <name_of_the_container>
 3. When you get to the docker's environment it will look like this: root@624f231882b2:/workspace#
 4. In the Docker terminal, use the source command to add environment variables:
  - root@624f231882b2:/workspace# source ./tpu-mlir/envsetup.sh

# Let's start converting the model to cvimodel format. *It is important not to compile cvite_tdl_sdk yet.
 1. Write this code on docker environment: 
      model_transform.py \
    --model_name yolov8s \
    --model_def yolov8s.onnx \ #your onnx model
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --test_input ./test.jpg \ #This is the image you copied into the docker environment. 
    --test_result yolov8s_top_outputs.npz \
    --mlir yolov8s.mlir
    
 After converting to the mlir file, a yolov9s_in_f32.npz file will be generated, which is the model's input file.
 
 2. Before quantizing to INT8 model, run calibration.py to get the calibration table.
    ```python
        run_calibration.py yolov8s.mlir \
        --dataset ./test \ #our test dataset that we copied into docker 
        --input_num 100 \
        -o yolov8s_cali_table
    ```

3. Then use the calibration table to generate the int8 symmetric cvimodel

    ```python
        model_deploy.py \
        --mlir yolov8s.mlir \
        --quant_input --quant_output \
        --quantize INT8 \
        --calibration_table yolov8s_cali_table \
        --chip cv181x \
        --test_input yolov8s_in_f32.npz \
        --test_reference yolov8s_top_outputs.npz \
        --tolerance 0.85,0.45 \
        --model yolov8s_cv181x_int8_sym.cvimodel
    ```
4. After compilation, a file named yolov8s_cv181x_int8_asym.cvimodel will be generated.

# Now that we have a model in .cvimodel format. We need to change the cvi_tdl configuration files. 
 1. Go to the docker directory on your local machine (mine is workspace) and find the file at this path cvitek_cvi_model/include/cvi_tdl/core/cvi_tdl_core.h. Open this file in your IDE
 2. Define there the variable 
    ```c 
        const char* model_path = “/home/max/workspace/yolov8_cv181x_int8_sym.cvimodel”; <path_to_your_cvimodel>
    ```
 3. Next on line 242 you will find the line DLL_EXPORT CVI_S32 CVI_TDL_OpenModel, replace the last argument with the path to the model you defined
    ```c
        DLL_EXPORT CVI_S32 CVI_TDL_OpenModel(cvitdl_handle_t handle, CVI_TDL_SUPPORTED_MODEL_E model, const char* model_path);
    ```
 4. After that, go back to the core directory where you found the cvi_tdl_core.h file before. And find the file cvi_tdl_custom.h. Define there a variable with the path to your cvi_model.
    ```c 
        const char* custom_path = “/home/max/workspace/yolov8_cv181x_int8_sym.cvimodel”; <path_to_your_cvimodel>
    ```
 5. On line 44, find DLL_EXPORT CVI_S32 CVI_TDL_Custom_SetModelPath and change the last argument to the path to your model 
    ```c 
        DLL_EXPORT CVI_S32 CVI_TDL_Custom_SetModelPath(cvitdl_handle_t handle, const uint32_t id, const char* custom_path);
    ```
# Next you need to change the configuration of the yolo_sample file
 1. You can find it at the following path: cvitek-tdl-sdk-sg200x/sample/cvi_yolo/sample_yolov8.cpp (open it in your IDE) 
 2. Define there the variable:
    ```c 
        const char* sample_path = “/home/max/workspace/yolov8_cv181x_int8_sym.cvimodel”; <path_to_your_cvimodel>
    ```
 3. On line 88, find ret = CVI_TDL_OpenModel, change the last argument to the path to your model:
    ```c 
        ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, const char* sample_path);
    ```
# Once you have changed all the configuration files, return to the docker environment and compile tdl_sdk
 1. Navigate to the host-tools folder 
   cd host-tools
   export PATH=$PATH:$(pwd)/gcc/riscv64-linux-musl-x86_64/bin
 2. Next, go to the cvitek_tdl_sdk directory and then to the ./sample directory and compile tdl_sdk
    cd cvitek-tdl-sdk-sg200x
    cd sample
    ./compile_sample.sh
 2. Now in the cvi_yolo folder (/workspace/cvitek-tdl-sdk-sg200x/sample/cvi_yolo/) you will find the file sample_yolov8. Copy this file to your milk_v duo board
    scp sample_yolov8 root@192.168.42.1:/root/
 3. Next, copy the file with your model in .cvimodel format and the image test.jpg
    scp  yolov8_cv181x_int8_sym.cvimodel test.jpg root@192.168.42.1:/root/
 4. After successfully copying the files, go to the milkv environment
    ssh root@192.168.42.1
    password: milkv
 5. And enter this in root to get the results
    ./sample_yolov8 ./yolov8_cv181x_int8_asym.cvimodel  test.jpg
    
   
  
    

   
   

 
 


 

 
 
    


