from __future__ import absolute_import, division, print_function, unicode_literals
from keras.applications.inception_v3 import preprocess_input as inception_preproc
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgba2rgb
import matplotlib.pyplot as plt
from keras import backend as K
from utils import load_image
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import saliency
import os


def extract_heatmaps(images_dir, heatmap_dir, num_of_images):
    """
        Sample images from the dataset and extract their heatmaps.

        Args:
            images_dir (str): path to dataset images that will be sampled
            heatmap_dir (str): path to store the heatmaps
            num_of_images (int): number of images to sample
    """
    sess = K.get_session()
    graph = sess.graph
    with graph.as_default():  # registers graph as default graph. Operations will be added to the graph
        model = InceptionV3(weights='imagenet')
        images = graph.get_tensor_by_name('input_1:0')
        logits = graph.get_tensor_by_name('predictions/Softmax:0')
        neuron_selector = tf.placeholder(tf.int32)  # Used to select the logit of the prediction
        y = logits[0][neuron_selector]  # logit of prediction
        prediction = tf.argmax(logits, 1)

        test_datagen = ImageDataGenerator(preprocessing_function=inception_preproc)
        test_generator = test_datagen.flow_from_directory(images_dir, target_size=(299, 299), batch_size=1, class_mode='categorical', shuffle=False)
        if num_of_images > len(test_generator.filenames):
            raise ValueError('The number of annotations cannot be higher than the number of available images.')
        image_selection = np.random.choice(len(test_generator.filenames), num_of_images)
        selected_images = np.array(test_generator.filenames)[image_selection]
        # Construct the saliency object.
        gradient_saliency = saliency.GradientSaliency(graph, sess, y, images)
        for i, img in enumerate(tqdm(selected_images)):
            # Create the folder if it does not exist
            if not os.path.exists(os.path.join(heatmap_dir, selected_images[i].split('/')[0])):
                os.makedirs(os.path.join(heatmap_dir, selected_images[i].split('/')[0]))
            # Skip if heatmap is already extracted
            if os.path.exists(os.path.join(heatmap_dir, selected_images[i])):
                continue
            im_int = load_image(os.path.join(images_dir, img))
            im = inception_preproc(im_int)
            # Predict label
            y_pred = sess.run(prediction, feed_dict={images: [im]})[0]
            # Compute the vanilla mask and the smoothed mask.
            smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, stdev_spread=.05, nsamples=10, feed_dict={neuron_selector: y_pred})
            # Call the visualization methods to convert the 3D tensors to 2D grayscale.
            smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
            # Extract images
            cm = plt.get_cmap('inferno')
            colored_heatmap = cm(smoothgrad_mask_grayscale)  # RGBA (A contains colormap) -> convert o RGB via rgba2rgb
            image_overlay = 0.5 * (im_int/255) + 0.5 * rgba2rgb(colored_heatmap)  # img1*alpha + img2*(1-alpha)
            plt.imsave(os.path.normpath(os.path.join(heatmap_dir + selected_images[i])), im_int)
            plt.imsave(os.path.normpath(os.path.join(heatmap_dir + selected_images[i][:-5] + '_heatmap' + selected_images[i][-5:])), image_overlay)


def extract_image_urls(cloud_bucket, heatmap_dir):
    """
        Sample images from the dataset and extract their heatmaps.

        Args:
            cloud_bucket (str): path of cloud storage used to host the heatmaps
            heatmap_dir (str): local path to the extracted heatmaps
    """
    class_folders = os.listdir(heatmap_dir)
    df_image_urls = pd.DataFrame(columns=["original_image_url", "heatmap_image_url"])
    image_count = 0  # number of images for which the image and heatmap name has been prossesed
    for class_name in class_folders:  # for each class of the current task
        if class_name == '.DS_Store':
            continue
        class_dir = os.path.normpath(os.path.join(heatmap_dir, class_name))
        class_heatmaps = os.listdir(class_dir)
        for index, image_name in enumerate(class_heatmaps):
            if image_name == '.DS_Store':
                continue
            if 'heatmap' not in image_name:  # if current file is an image
                df_image_urls.loc[image_count, 'original_image_url'] = cloud_bucket + class_name + '/' + image_name
                df_image_urls.loc[image_count, 'heatmap_image_url'] = cloud_bucket + class_name + '/' + image_name[:-5] + '_heatmap' + image_name[-5:]
            image_count += 1
    df_image_urls.to_csv(os.path.normpath(os.path.join('./local_interpretability_extraction', 'mturk_input.csv')), index=False)

