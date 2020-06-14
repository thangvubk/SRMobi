package com.example.srandroidapp;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;


public class InferenceEngine {

    public enum Device {
        CPU, GPU
    }

    private String modelFileName;
    private Interpreter model;
    private int[] inputShape;
    private int[] outputShape;
    private DataType dType = null;

    public Interpreter getModel() {
        return model;
    }

    public int[] getInputShape() {
        return inputShape;
    }

    public int[] getOutputShape() {
        return outputShape;
    }

    public DataType getDataType() {
        return dType;
    }

    public InferenceEngine(AssetManager assetManager, String modelFileName, Device device,
                           int numThreads) {
        MappedByteBuffer modelMapBuffer = loadModelFile(assetManager, modelFileName);
        assert modelMapBuffer != null : "Fail to load model";
        Interpreter.Options options = new Interpreter.Options();
        switch (device) {
            case GPU:
                GpuDelegate gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
            case CPU:
                break;
        }
        options.setNumThreads(numThreads);
        model = new Interpreter(modelMapBuffer, options);
        inputShape = model.getInputTensor(0).shape();
        outputShape = model.getOutputTensor(0).shape();
        dType = model.getInputTensor(0).dataType();
    }

    public TensorBuffer inference(TensorImage lrImg) {
        assert lrImg != null;
        TensorBuffer srImg = TensorBuffer.createFixedSize(outputShape, dType);
        model.run(lrImg.getBuffer(), srImg.getBuffer().rewind());
        return srImg;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelFileName) {
        try {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(modelFileName);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
