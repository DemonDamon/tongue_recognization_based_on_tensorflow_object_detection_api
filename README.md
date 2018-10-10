# tongue_recognization
 - Fine-tune models like ssd_mobilenet、faster_rcnn models in COCO dataset for detecting tongues
 - All elementary codes are from TensorFlow Object Detection API
 - The Chinese version tutorial of my little project is here https://mp.weixin.qq.com/s?__biz=MzIyMjE5Njk1Mw==&mid=2651247687&idx=1&sn=737d94e4503d1bc0585137aea49f922d&scene=19#wechat_redirect
 - 'tongue_inference_graph_latest.pb' is the latest model I trained, so you can use it for tongue detection directly
 - the folders 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017' and 'ssd_mobilenet_v1_coco_11_06_2017' are empty, you have
   to download the inference graph respectively into these folders 
 - The dataset is little, you can follow the link above to enrich your dataset by labelImge tool, then rebuild .csv and .record files
 
DevEvn `Python-v3(3.6)`：

 - tensorflow==1.9.0

Hardware Equipment
 - 4 core CPU
 - 24gb memory
 - 11gb video memory
 - 484gb/s bandwidth of video memory 
 - 11.34tflop (single precision)
 - 2T hard disk

Result
  - ![image](https://github.com/DemonDamon/tongue_recognization/blob/master/test_tongue_1.png)
  - ![image](https://github.com/DemonDamon/tongue_recognization/blob/master/test_tongue_2.png)

