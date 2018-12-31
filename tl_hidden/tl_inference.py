from tl_hidden.utils import load_images

import tensorflow as tf
import numpy as np

def extract_hidden_features_from_disk(input_tensor, 
                                      output_tensor, 
                                      image_shape, 
                                      list_images,
                                      path,
                                      batch_img_size = 1024*10,
                                      batch_tf_size = 1024,
                                      padding = True
                                     ):
    
    batches_img_no = int(len(list_images) / batch_img_size) + 1
    image_names = []
    hidden_features = np.empty((0, 2048))
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
    
        for batch_img in range(0, batches_img_no):
            batch_img_min = batch_img_size*batch_img
            batch_img_max = min((batch_img_size*(batch_img+1)), len(list_images))
            print('Batch images:' + str(batch_img))
            images_batch = load_images(path, list_images[batch_img_min:batch_img_max], image_shape = image_shape, padding = padding)
            print(images_batch.shape)
            image_names_batch = list_images[batch_img_min:batch_img_max]
            
            batches_tf_no = int((batch_img_max-batch_img_min) / batch_tf_size) + 1
        
            for batch_tf in range(0, batches_tf_no):
                print('Batch tensorflow:' + str(batch_tf))
                batch_tf_min = batch_tf_size*batch_tf
                batch_tf_max = min(batch_tf_size*(batch_tf+1), batch_img_size)
                images_tf = images_batch[batch_tf_min:batch_tf_max]
                names_tf = image_names_batch[batch_tf_min:batch_tf_max]
            
                output = session.run(output_tensor, feed_dict={input_tensor: images_tf})
            
                hidden_features = np.concatenate((hidden_features, output), axis = 0)
                image_names = image_names + names_tf
    
    return(hidden_features, image_names)