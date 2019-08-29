# Keras_amp

Steps to run the example

## 1 -- clone repository

```bash
git clone  https://github.com/DEKHTIARJonathan/KerasLossScaleOptimizer_Demo.git && cd KerasLossScaleOptimizer_Demo
```  

## 2. download data and put the files in your directory

Download Link: https://drive.google.com/drive/folders/1ryQGhurGivPV-N-16m_jPIpnedfm4QR0?usp=sharing

## 3 -- build container

```bash
docker build -t keras_lossscaleoptimizer_demo .
```  

## 4 -- launch/run the container

```bash
docker run --runtime=nvidia -it --rm \
    --network=host \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -v $(pwd):/project \
    keras_lossscaleoptimizer_demo bash
```
    
## 5 Execute the python script

```bash
./update_keras.sh && python keras_amp_example.py
``` 
