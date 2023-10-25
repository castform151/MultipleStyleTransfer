# Multiple Style Transfer
This work is based on this paper [Multiple Style Transfer via Variational AutoEncoder
](https://arxiv.org/abs/2110.07375)
     
## For proposed ST-VAE model, we claim the following points:

• First working on using Variational AutoEncoder for image style transfer.

• Multiple style transfer by proposed VAE based Linear Transformation.


## Complete Architecture
The complete architecture is shown as follows,

![network](/figure/figure1.PNG)

## Set-up
1. Download pre-trained models from [here](https://drive.google.com/file/d/1WZrvjCGBO1mpggkdJiaw8jp-6ywbXn4J/view?usp=sharing) and copy them to the folder `models/`

2. Put your content images under `Test/content` and your style images under `Test/style`. Sample content and style images are available in `Test/extra_style` and `Test/extra_content` folders.

3. Run `python3 requirements.txt` to get all pre-requsites
