package org.example;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

public class Demo {

    private static final Logger log = LoggerFactory.getLogger(Demo.class);

    public static void main(String[] args) {
        // See : https://github.com/jcuda/jcuda-samples/blob/master/JCudaSamples/src/main/java/jcuda/runtime/samples/JCudaPrintDeviceInfo.java
        JCuda.setExceptionsEnabled(true);
        int[] deviceCount = {0};
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
            System.out.println("Number of multiprocessors on device = " + deviceProperties.multiProcessorCount);
            System.out.println("Nb cores = " + getSPcores(deviceProperties));
        }
    }

    /**
     * Inspired from https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
     */
    public static int getSPcores(cudaDeviceProp devProp) {
        int cores = 0;
        int mp = devProp.multiProcessorCount;
        switch (devProp.major) {
            case 2 -> {
                if (devProp.minor == 1) {
                    cores = mp * 48;
                } else {
                    cores = mp * 32;
                } // Fermi
            }
            case 3 -> // Kepler
                    cores = mp * 192;
            case 5 -> // Maxwell
                    cores = mp * 128;
            case 6 -> {
                if ((devProp.minor == 1) || (devProp.minor == 2)) {
                    cores = mp * 128;
                } else if (devProp.minor == 0) {
                    cores = mp * 64;
                } else {
                    System.out.println("Unknown device type");
                } // Pascal
            }
            case 7 -> {
                if ((devProp.minor == 0) || (devProp.minor == 5)) {
                    cores = mp * 64;
                } else {
                    System.out.println("Unknown device type");
                } // Volta and Turing
            }
            case 8 -> {
                if (devProp.minor == 0) {
                    cores = mp * 64;
                } else if (devProp.minor == 6) {
                    cores = mp * 128;
                } else if (devProp.minor == 9) {
                    cores = mp * 128; // ada lovelace
                } else {
                    System.out.println("Unknown device type");
                } // Ampere
            }
            case 9 -> {
                if (devProp.minor == 0) {
                    cores = mp * 128;
                } else {
                    System.out.println("Unknown device type");
                } // Hopper
            }
            default -> System.out.println("Unknown device type");
        }
        return cores;
    }

}
