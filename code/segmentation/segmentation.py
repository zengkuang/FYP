import sys
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy import misc
import tensorflow as tf
import get_dataset_colormap
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)
class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""
	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	def __init__(self, model_path):
	    """Creates and loads pretrained deeplab model."""
	    self.graph = tf.Graph()
	    with open(model_path, 'rb') as fd:
	        graph_def = tf.GraphDef.FromString(fd.read())
	    with self.graph.as_default():
	        tf.import_graph_def(graph_def, name='')
	    self.sess = tf.Session(graph=self.graph)
	def run(self, image):
		"""Runs inference on a single image.
	    Args:
	        image: A PIL.Image object, raw input image.
    	Returns:
    	    resized_image: RGB image resized from original input image.
    	    seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={
			self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map

def vis_segmentation(image, seg_map, img_name):
	plt.figure()
	plt.subplot(221)
	plt.imshow(image)
	plt.axis('off')
	plt.title('input image')
	plt.subplot(222)
	seg_image = get_dataset_colormap.label_to_color_image(
	seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    #print(seg_image.shape)
    #plt.imshow(seg_image(seg_image.shape[0],seg_image.shape[1]))
    #make all non zero pixel 1
	bimap = np.equal(seg_map,0) + 0
	plt.imshow( bimap, cmap = plt.cm.binary)
	plt.axis('off')
	plt.title('segmentation map')
	plt.subplot(223)
	#plt.imshow(image)
	#plt.imshow(seg_image, alpha=0.7) 
	backgrounds = np.multiply(image, np.expand_dims(bimap,axis = 2))
	plt.imshow(backgrounds)
	plt.axis('off')
	plt.title('background segmentation')
	unique_labels = np.unique(seg_map)
	ax = plt.subplot(224)
	foregrounds = np.multiply(image, np.expand_dims(1-bimap,axis = 2))
	plt.imshow(foregrounds)
	plt.axis('off')
	plt.title('foreground segmentation')
	unique_labels = np.unique(seg_map)
	#bimap --> 1 denotes background(white) and 0 denotes foreground(black)
	#the result of seg map is inversed for better visualization
	'''plt.imshow(
	    FULL_COLOR_MAP[unique_labels].astype(np.uint8),
	    interpolation='nearest')
	ax.yaxis.tick_right()
	plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
	plt.xticks([], [])
	ax.tick_params(width=0)
	'''
	misc.imsave('../../sample_img/{}_alpha.png'.format(img_name), 1-bimap)
	misc.imsave('../../sample_img/{}_rgb.png'.format(img_name), image) 
	plt.show()

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Usage: python {} image_path model_path'.format(sys.argv[0]))
		exit()
	image_path = sys.argv[1]
	model_path = sys.argv[2]
	model = DeepLabModel(model_path)
	orignal_im = Image.open(image_path)
	resized_im, seg_map = model.run(orignal_im)
	print(image_path.split('/'))
	img_name = image_path.split('/')[3].split('.')[0]
	vis_segmentation(resized_im, seg_map, img_name)



