# LicensePlate_Yolov8_FilterCNN_PaddleOCR
Project that uses Yolov8 as license plate detector, followed by a filter that is got selecting from a filters collection with a code assigned to each filter and predicting what filter with a CNN process


Download all the files of the project to a folder, unzip the zip files.

All the modules necessary for its execution can be installed, if the programs give a module not found error, by means of a simple pip.

The most important:

paddleocr must be installed (https://pypi.org/project/paddleocr/)

pip install paddleocr

yolo must be installed, if not, follow the instructions indicated in: https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

pip install ultralytics

Are attached, te best.pt file, that allows the license plate detect for yolo,  and FSRCNN that allows the working of the filter with de same name

As a previous step, the X_train and the Y_train that the CNN needs are created, the X_Train is the matrix of each image and the Y_train is made based on the code assigned (from 0 to 10) to the first filter with which paddleocr manages to recognize that license plate of car in the reference project https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR.

The Crea_Xtrain_Ytrain.py program is attached (its execution is not necessary), whose result after applying it to different image files (the input file is indicated in instruction 15) of renamed image cars with their registration plate is saved in the Training folder , consisting of the image itself and a .txt file with the name of the car's license plate and containing the filter code

assigned to that image. This Training file that is attached in .zip format is necessary to download it, like the Test.zip file and unzip. 
You may run

TrainCodFilterCNN.py

but is not necessary because  the files with the weights and the necessary and verified information are attached: FilterCNN16Hits500epoch.json and FilterCNN16Hits500epoch.h5. Since the CNN training process can give different results in each execution of the training.

So you can run directly

GetNumberInternationalLicensePlate_Yolov8_CNNFilters_PaddleOCR_V1.py

Any folder may be tested changing instruction 14, the resultas in LicenseResults.txt file

Comparing with the reference project: https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR you get a lower precision but a considerable reduction in execution time.

The references are identical to those of the project https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR:


References:

https://pypi.org/project/paddleocr/

https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

https://public.roboflow.com/object-detection/license-plates-us-eu/3

https://docs.ultralytics.com/python/

https://medium.com/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c

https://medium.com/@alimustoofaa/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6

https://medium.com/adevinta-tech-blog/text-in-image-2-0-improving-ocr-service-with-paddleocr-61614c886f93

https://machinelearningprojects.net/number-plate-detection-using-yolov7/

https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters

https://github.com/mrzaizai2k/VIETNAMESE_LICENSE_PLATE

https: //www.doubango.org/webapps/alpr/

Filters:

https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models ( downloaded module FSRCNN_x4.pb)

https://learnopencv.com/super-resolution-in-opencv/#sec5

https://learnopencv.com/super-resolution-in-opencv/

https://gist.github.com/endolith/334196bac1cac45a4893#

https://stackoverflow.com/questions/46084476/radon-transformation-in-python

https://gist.github.com/endolith/255291#file-parabolic-py

https://learnopencv.com/otsu-thresholding-with-opencv/ 

https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45

https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05

https://programmerclick.com/article/89421544914/

https://anishgupta1005.medium.com/building-an-optical-character-recognizer-in-python-bbd09edfe438

https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/

https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e

https://github.com/victorgzv/Lighting-correction-with-OpenCV

https://medium.com/@yyuanli19/using-mnist-to-visualize-basic-conv-filtering-95d24679643e
