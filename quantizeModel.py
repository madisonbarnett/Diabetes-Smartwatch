import tensorflow as tf 

model = tf.keras.models.load_model('./model_weights/lstm_model2_5s_nonlin.keras')
print("Model loaded. On to quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open('./model_weights/lstm_model2_5s_nonlin_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Successfully saved quantized model to './model_weights/lstm_model2_5s_nonlin_quantized.tflite'")