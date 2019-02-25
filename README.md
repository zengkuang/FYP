# FYP

Hackers, please add your model under ./code and create our own folder for the very model. For example, if wish to add a image inpainting model, name it ./code/non_learning_inpainting.

Testing images are located under ./sample_img. There are few to select from. Feel free to push any other images as possible and name it in the order of "image1", "image2" etc. 



Convolutional AutoEncoder Trained for MNIST dataset has been uploaded. It super strong in terms of MNist dataset.
The model will further be trained for inpainting purpose, and serve as an function to extract high-level spatial feature for the input conditional image of CDCGAN.

Here are some training results of CAE: https://hkustconnect-my.sharepoint.com/personal/lchenbg_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flchenbg_connect_ust_hk%2FDocuments%2FAttachments%2FFYP%2Ezip&parent=%2Fpersonal%2Flchenbg_connect_ust_hk%2FDocuments%2FAttachments
(note: fig 5000 above are reduntant figures. The model does not improve but overfits after it. The test result of reconsturction can be seen in figure infer.jpg)

MNIST Inpainting dataset has been uploaded to https://hkustconnect-my.sharepoint.com/personal/lchenbg_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flchenbg%5Fconnect%5Fust%5Fhk%2FDocuments%2FAttachments%2Fmnist%5Finpainting%2Enpz&parent=%2Fpersonal%2Flchenbg%5Fconnect%5Fust%5Fhk%2FDocuments%2FAttachments\

usage: \
$data = numpy.load("mnist_inpainting.npz")$\
$train_X = data['x']$\
$train_y = data['y']$\

X contains the corrupted image and y contains the original one.\

CDCGAN model will be tested using this data first, then further developed to nature images.\

DCGAN results available here: https://hkustconnect-my.sharepoint.com/personal/lchenbg_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flchenbg_connect_ust_hk%2FDocuments%2FAttachments%2Ffigs%2Ezip&parent=%2Fpersonal%2Flchenbg_connect_ust_hk%2FDocuments%2FAttachments\

Nature image inpainting can be found here: 

The networkw works well on MNIST data set.\

Future development plan:\
~ end of Feb => Done model training on MNIST, prototype of close-form matting/ inpainting\
March ~ April => Tune models on Nature Image\
Apirl ~ May => Done model training on nature images. GUI developement\

