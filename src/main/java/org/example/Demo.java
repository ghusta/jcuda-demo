package org.example;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

public class Demo {

    public static void main(String[] args) {
        JCuda.setExceptionsEnabled(true);
        int deviceCount[] = {0};
        cudaGetDeviceCount(deviceCount);
        System.out.println("Found " + deviceCount[0] + " devices");
        for (int device = 0; device < deviceCount[0]; device++) {
            System.out.println("Properties of device " + device + ":");
            cudaDeviceProp deviceProperties = new cudaDeviceProp();
            cudaGetDeviceProperties(deviceProperties, device);
            System.out.println(deviceProperties.toFormattedString());

            // https://developer.nvidia.com/cuda-gpus
            System.out.printf("GPU Compute Capability : %d.%d %n", deviceProperties.major, deviceProperties.minor);
            System.out.println("Device name : " + deviceProperties.getName());
            System.out.println("Total global memory = " + Math.round(deviceProperties.totalGlobalMem / (1024 * 1024 * 1024.0)) + " GB");
        }
    }

}
