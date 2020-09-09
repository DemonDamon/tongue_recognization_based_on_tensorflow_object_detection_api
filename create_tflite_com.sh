cd ./tongue_recognization_based_on_tensorflow_object_detection_api-master

export PYTHONPATH=$PYTHONPATH:'pwd':'pwd'/slim

python object_detection/export_tflite_ssd_graph.py --pipeline_config_path=saved_model/depth_multiplier_0.6/pipeline.config --trained_checkpoint_prefix=saved_model/depth_multiplier_0.6/model.ckpt-20000 --output_directory=export_tongue_inference_ssdlite_mobilenetv2/depth_multiplier_0.6/ --add_postprocessing_op=true

git clone https://github.com/tensorflow/tensorflow.git

cd ./tensorflow-master

bazel run -c opt tensorflow/lite/toco:toco -- --input_file=./tongue_recognization_based_on_tensorflow_object_detection_api-master/export_tongue_inference_ssdlite_mobilenetv2/depth_multiplier_0.6/tflite_graph.pb --output_file=./tongue_recognization_based_on_tensorflow_object_detection_api-master/export_tongue_inference_ssdlite_mobilenetv2/depth_multiplier_0.6/tongue_detect_0.6.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  --inference_type=FLOAT --allow_custom_ops
