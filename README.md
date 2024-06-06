## Dependencies

* Python 3.9

## Setting up  conda Environment

* Clone the repository by running 
```
git clone https://github.com/hasu234/Image_Classifier.git
```
* Change the current directory to Image_Classifier 
```
cd Image_Classifier
```
* Create a conda environment 
```
conda create -n myenv python=3.9
```
* Activate the environment 
```
conda activate myenv
```
* Install the required library from ```requirment.txt``` by running 
```
pip install -r requirmen.txt
```
* Or, create a conda environment from ```environment.yml``` by running 
```
conda env create -f environment.yml
```

## Training your data
* To train on your dataset
Make sure you have a data folder having the same folder hierarchy like below
```
├── dataset
|   ├── train
│   │   ├── class1
│   │   │   ├──image1.jpg
│   │   │   ├──image2.jpg
│   │   ├── class2
│   │   │   ├──image1.jpg
│   │   │   ├──image2.jpg
│   │   ├── class3
│   │   │   ├──image1.jpg
│   │   │   ├──image2.jpg
│   │   ├── class4
│   │   │   ├──image1.jpg
│   │   │   ├──image2.jpg
|   ├── test
│   │   ├── class1
│   │   │   ├──image1.jpg
│   │   │   ├──image2.jpg
│   │   ├── class2
│   │   │   ├──image1.jpg
│   │   │   ├──image2.jpg
│   │   ├── class3
│   │   │   ├──image1.jpg
│   │   │   ├──image2.jpg
│   │   ├── class4
│   │   │   ├──image1.jpg
│   │   │   ├──image2.jpg
```
or make some changes on ```train.py``` according to your dataset directory.
* Make sure you are in the project directory and run the ```train.py``` script with the folder directory of your dataset
```
python train.py /path/to/dataset_directory
```
## Running inference
* To run the inference on your test data make sure you downloaded the pre-trained model weight from [this link](https://drive.google.com/uc?id=197Kuuo4LhHunYLgGKfGeouNTL0WguP0T&export=download).
* Then run the ```infer.py``` script from the terminal specifying the test image location and downloaded pre-trained model location
```
python infer.py path/to/image.jpg path/to/model.pth
```


## Running on Docker
* Clone the repository by running 
```
git clone https://github.com/hasu234/Image_Classifier.git
```
* Change the current directory to Image_Classifier 
```
cd Image_Classifier
```
* Build the Docker image by running 
```
docker build -t sdpdsample .
```
* Run the docker image 
```
docker run -d sdpdsample
```
if the container fails to run in the background, run it in the foreground using ```docker run -it sdpdsample``` then exit to get the running container id
* Get the container ID
```
docker ps
```
* Getting inside the container
```
docker exec -it <container id> bash
```
You will get a Linux-like command-line interface
* Running the project
```
# For training on your data
python train.py /path/to/dataset_directory

# for running inference
python infer.py path/to/image.jpg path/to/model.pth
```
