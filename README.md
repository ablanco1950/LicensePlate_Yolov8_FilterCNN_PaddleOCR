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
