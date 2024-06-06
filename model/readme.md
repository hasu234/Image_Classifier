# Download Pretraied Model

The model is being saved using ```torch.save(model.state_dict(), 'resnet_model.pth')```, so if you want to load the model outside this repository make sure to create an instance of the model architecture from [here](https://github.com/hasu234/Image_Classifier/blob/main/classifier/Resnet.py) and then load the model using ```model.load_state_dict(torch.load(model_path, map_location=torch))```.
