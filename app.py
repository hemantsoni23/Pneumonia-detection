# # import numpy as np
# # import streamlit as st
# # import cv2
# # import tensorflow as tf
# # from keras.applications.densenet import preprocess_input
# #
# # def preprocess_image(image):
# #     img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     img = cv2.resize(img, (224, 224))
# #     return img
# #
# # def main():
# #     st.title("Pneumonia Detection App")
# #     st.write("Upload an image to check for pneumonia.")
# #
# #     # Upload image
# #     uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
# #
# #     if uploaded_image is not None:
# #         # Preprocess and display uploaded image
# #         # Preprocess and display uploaded image
# #         image = preprocess_image(uploaded_image)
# #         image = image / 255.0
# #         st.image(image, caption='Uploaded Image', use_column_width=True)
# #
# #         # Perform pneumonia detection
# #         prediction = model.predict(image[np.newaxis, ...])[0, 0]
# #
# #         if prediction > 0.5:
# #             result = 'Pneumonia'
# #         else:
# #             result = 'Normal'
# #
# #         st.write(f"Prediction: {result} ({prediction:.2f})")
# #
# #
# # if __name__ == '__main__':
# #     # Load the trained model
# #     model = tf.keras.models.load_model('model.h5')
# #
# #     # Run the Streamlit app
# #     main()
# #
# import numpy as np
# import streamlit as st
# import cv2
# import tensorflow as tf
# from keras.applications.densenet import preprocess_input
# from tensorflow.keras.models import Model
#
# def preprocess_image(image):
#     img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))
#     return img
#
# def generate_heatmap(model, image):
#     # Preprocess the image
#     preprocessed_image = preprocess_input(image)
#
#     # Expand dimensions to match model input shape
#     preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
#
#     # Get the target convolutional layer for heatmap generation
#     target_layer = model.get_layer("conv5_block16_concat")
#
#     # Create a new model that outputs the target layer's activation and the model's prediction
#     heatmap_model = Model(inputs=model.input, outputs=[target_layer.output, model.output])
#
#     # Compute the gradient tape
#     with tf.GradientTape() as tape:
#         conv_output, predictions = heatmap_model(preprocessed_image)
#         target_class = predictions[:, 0]  # Assuming binary classification
#
#     # Get the gradients of the target class with respect to the output feature map
#     grads = tape.gradient(target_class, conv_output)
#
#     # Compute the channel-wise mean of the gradients
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#
#     # Reshape pooled_grads to match the shape of conv_output
#     pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))
#
#     # Multiply each channel in the feature map by the corresponding gradient value
#     heatmap = tf.reduce_mean(conv_output * pooled_grads, axis=-1)
#
#     # Normalize the heatmap
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
#
#     return heatmap
#
#
#
# def main():
#     st.title("Pneumonia Detection App")
#     st.write("Upload an image to check for pneumonia.")
#
#     # Upload image
#     uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
#
#     if uploaded_image is not None:
#         # Preprocess and display uploaded image
#         image = preprocess_image(uploaded_image)
#         image = image / 255.0
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#
#         # Perform pneumonia detection
#         prediction = model.predict(image[np.newaxis, ...])[0, 0]
#
#         if prediction > 0.5:
#             result = 'Pneumonia'
#         else:
#             result = 'Normal'
#
#         st.write(f"Prediction: {result} ({prediction:.2f})")
#
#         # Generate and display heatmap
#         heatmap = generate_heatmap(model, image)
#         heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
#         heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#         st.image(heatmap, caption='Heatmap', use_column_width=True)
#
# if __name__ == '__main__':
#     # Load the trained model
#     model = tf.keras.models.load_model('model.h5')
#
#     # Run the Streamlit app
#     main()

# import numpy as np
# import streamlit as st
# import cv2
# import tensorflow as tf
# from keras.applications.densenet import preprocess_input
# from tensorflow.keras.models import Model
# import json
# from streamlit_lottie import st_lottie,st_lottie_spinner
#
# with open("animation2.json", 'r') as f:
#     animation_frames = json.load(f)
#
# theme = {
#     "primaryColor": "#FF4B4B",
#     "backgroundColor": "#0E1117",
#     "secondaryBackgroundColor": "#262730",
#     "textColor": "#FAFAFA",
#     "font": "sans-serif"
# }
# st.set_page_config(page_title="Pneumonia Detection", page_icon=":smiley:", layout="wide",
#                    initial_sidebar_state="collapsed")
#
# def preprocess_image(image):
#     img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))
#     return img
#
# def generate_heatmap(model, image):
#     preprocessed_image = preprocess_input(image)
#     preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
#     target_layer = model.get_layer("conv5_block16_concat")
#     heatmap_model = Model(inputs=model.input, outputs=[target_layer.output, model.output])
#
#     # Compute the gradient tape
#     with tf.GradientTape() as tape:
#         conv_output, predictions = heatmap_model(preprocessed_image)
#         target_class = predictions[:, 0]  # Assuming binary classification
#
#     # Get the gradients of the target class with respect to the output feature map
#     grads = tape.gradient(target_class, conv_output)
#
#     # Compute the channel-wise mean of the gradients
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#
#     # Reshape pooled_grads to match the shape of conv_output
#     pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))
#
#     # Multiply each channel in the feature map by the corresponding gradient value
#     heatmap = tf.reduce_mean(conv_output * pooled_grads, axis=-1)
#
#     # Normalize the heatmap
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
#
#     return heatmap
#
# def main():
#     st.title("Pneumonia Detection App")
#     st.write("Upload an image to check for pneumonia.")
#     uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
#
#     if uploaded_image is not None:
#         with st_lottie_spinner(animation_frames, height=500, width=500, key='main', quality='high'):
#           image = preprocess_image(uploaded_image)
#           image = image / 255.0
#           st.image(image, caption='Uploaded Image', width=400)
#           prediction = model.predict(image[np.newaxis, ...])[0, 0]
#           if prediction > 0.5:
#               result = 'Pneumonia'
#           else:
#               result = 'Normal'
#
#         st.write(f"Prediction: {result} ({prediction:.2f})")
#
#
# if __name__ == '__main__':
#     model = tf.keras.models.load_model('model.h5')
#     main()
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
import json
import time
from streamlit_lottie import st_lottie_spinner

with open("animation2.json", 'r') as f:
    animation_frames = json.load(f)

theme = {
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#FAFAFA",
    "font": "sans-serif"
}
st.set_page_config(page_title="Pneumonia Detection", page_icon=":smiley:", layout="wide",
                   initial_sidebar_state="collapsed")


def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img


def main():
    st.title("Pneumonia Detection App")
    st.write("Upload an image to check for pneumonia.")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            image = preprocess_image(uploaded_image)
            image = image / 255.0
            st.image(image, caption='Uploaded Image', width=400)

        with col2:
            with st_lottie_spinner(animation_frames,height=400,width=400,key='animation_key'):
                prediction = model.predict(image[np.newaxis, ...])[0, 0]
                if prediction > 0.5:
                    result = 'Pneumonia'
                else:
                    result = 'Normal'
                time.sleep(0)
            # st.write(f"Detection: {result} ({prediction:.2f})")
            st.markdown(f"##   Detection: {result} ({prediction:.2f})")  # Detection output line


if __name__ == '__main__':
    model = tf.keras.models.load_model('model.h5')
    main()



