package com.example.srandroidapp;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


public class MainActivity extends AppCompatActivity {
    private String modelFileName = "model.tflite";
    private String imgFileName = "butterfly.png";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ImageView imgViewLR = (ImageView) findViewById(R.id.LRImg);
        ImageView imgViewSR = (ImageView) findViewById(R.id.SRImg);

        // Load img, model and perform inference
        AssetManager assetManager = getAssets();
        InferenceEngine inferenceEngine = new InferenceEngine(assetManager, modelFileName,
                InferenceEngine.Device.CPU, 1);
        DataType dtype = inferenceEngine.getDataType();
        int[] inputShape = inferenceEngine.getInputShape();
        TensorImage lrImg = Image.loadImage(assetManager, imgFileName, inputShape, dtype);
        TensorBuffer srImg = inferenceEngine.inference(lrImg);

        // show lr, sr images
        Bitmap lrImgBitmap = lrImg.getBitmap();
        int[] outputShape = inferenceEngine.getOutputShape();
        Bitmap srImgBitmap = Image.convertTensorBufferToBitmap(srImg, outputShape);
        imgViewLR.setImageBitmap(lrImgBitmap);
        imgViewSR.setImageBitmap(srImgBitmap);
    }
}