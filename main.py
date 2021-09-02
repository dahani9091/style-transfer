
import mxnet as mx
import numpy as np
import net
import utils
import streamlit as st
from PIL import Image
import os


def evaluate(content_image,style_image,output_image,model='models/21styles.params',style_size=600,content_size=600,cuda=0):
    if cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    # images
    content_image = utils.tensor_load_rgbimage(content_image,ctx, size=content_size, keep_asp=True)
    style_image = utils.tensor_load_rgbimage(style_image, ctx,size=style_size)
    style_image = utils.preprocess_batch(style_image)
    # model
    style_model = net.Net(ngf=128)
    style_model.load_parameters(model, ctx=ctx)
    # forward
    style_model.set_target(style_image)
    output = style_model(content_image)
    utils.tensor_save_bgrimage(output[0], output_image, cuda)




if __name__ == '__main__':
    st.set_page_config(page_title='IA ART', page_icon = 'pencil.jpg', layout = 'wide', initial_sidebar_state = 'auto')

    ### Side bar  ###########
    all_style_img = [img_.replace('.jpg','') for img_ in os.listdir(os.path.join(os.getcwd(),'images','styles'))]
    style_name = st.sidebar.selectbox(
        'Select Style',
        tuple(all_style_img)
    )

    ### Body ##############

    

    # show style image
    style_image_path  = os.path.join(os.getcwd(),'images','styles',style_name+'.jpg')
    st.subheader("Style image :")
    style_image = Image.open(style_image_path)
    st.image(style_image, width=500) # image: numpy array


    ## get input image and show it
    st.subheader("Origin image :")
    #input_image_file = st.file_uploader("Upload your Image",type=['png','jpeg','jpg'])
    image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    
    if image_file != None :
        # show input image
        input_image = image_file
        st.image(input_image, width=700) # image: numpy array
        ## output path
        output_image ='/tmp/'+'out'+'.jpg' 

        # start precess
        clicked = st.button('Stylize')

        if clicked:
            #model = style.load_model(model)

            #style.stylize(model, input_image, output_image)
            evaluate(input_image,style_image_path,output_image)
            st.subheader("Output image: ")
            image = Image.open(output_image)
            st.image(image, width=500)



