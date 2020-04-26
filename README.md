# Pix2Pix with Plans
### Generation of Floorplans with Pix2Pix for conversion in 3D models - PyTorch
## Requirements
Install requirements found in requirements.txt by running ```pip3 install -r requirements.txt```
## Dataset
Put your plans in dataset/ , run ```python3 image_transformer.py``` to process the data and create shapes. If already in lines(walls+windows/doors) and shapes, put directly under dataset/images/ following the same structure compatible with Torchvision data loader.
## Training of Pix2Pix
To train the Pix2pix run ```python3 train.py```
Change the options for training wanted in options/parse.py, e.g. batch size, condtional or not, learning rate, version of generator ...
The checkpoints for generator and discriminator, with samples generated are saved under temp/ . Import new checkpoints by modifying options and placing them under dcgan/checkpoints. 
## Processing with RTV (SEG + IP)
Run ```python3 eval.py``` to evaluate RTV(Segmentation Network + IP-fusion) on inputs placed manually in rtv_inputs, outputs are saved in rtv_outputs. It allows for tuning of IP hyperparameters. Under IP_masks/ can be found IP_heatmaps which allow processing of heatmaps with IP and view of current heatmaps, and RTV_heatmaps which shows result of Data transformer from RTV paper. 
## Generation
When the training is complete and IP parameters tuned, one can generate new floorplans based on shapes of floorplans put in inputs/ if conditional or number of wanted free generations by specifying a number if not conditional. Run ```python3 generate.py``` Use latest checkpoint of generator to generate. 
## Conversion to 3D models 
The resulting outputs are saved under outputs/, with the samples before processing of RTV and after processing under outputs/rtv/ . The txt files can be found to reconstruct new 3D floorplan models in Revit. 

