package com.example.srandroidapp;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.core.math.MathUtils;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class Image {

    public static TensorImage loadImage(AssetManager assetManager, String imgFilename,
                                        int[] shape, DataType dType) {
        int height = shape[1];
        int width = shape[2];
        InputStream inputStream;
        try {
            inputStream = assetManager.open(imgFilename);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        Bitmap imgBitmap = BitmapFactory.decodeStream(inputStream);
        TensorImage imgBuffer = new TensorImage(dType);
        imgBuffer.load(imgBitmap);

        // Create processor for TensorImage, image transforms can be added here
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(height, width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .build();
        return imageProcessor.process(imgBuffer);
    }

    public static Bitmap convertTensorBufferToBitmap(TensorBuffer tensorBuffer, int[] shape) {
        int height = shape[1];
        int width = shape[2];
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        ByteBuffer buffer = tensorBuffer.getBuffer();
        buffer.rewind();
        assert buffer.remaining() == 4 * width * height;  // a float value is 4 bytes

        int[] pixels = new int[width * height];
        for (int i = 0; i < width * height; i++) {
            int a = 255;
            int r = Math.round(MathUtils.clamp(buffer.getFloat(), 0, 255));
            int g = Math.round(MathUtils.clamp(buffer.getFloat(), 0, 255));
            int b = Math.round(MathUtils.clamp(buffer.getFloat(), 0, 255));
            pixels[i] = (a << 24) + (r << 16) + (g << 8) + b;
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }
}
