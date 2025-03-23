
Aaltoes CV_hackathon v1. 21/03/2025 - 23/03/2025

Team: Irem Sahin, Rishika Shivakumar, Jamila Shulekha

Challenge: 
"Welcome to the AaltoES 2025 Computer Vision Hackathon! In this competition, your goal is to develop models that can accurately detect inpainted (manipulated) regions in images. This binary segmentation task has important applications in digital forensics, media authentication, and combating misinformation.
Segment regions in images that have been manipulated through ai-generated techniques. This is a binary segmentation task where each pixel must be classified as either real (0) or fake (1)."

We created an ML model based on the UNET framework to tackle this challenge over two days, our model currently runs with a dice score of 0.8164 and scored 0.58124 in the Kaggle competition. It has been trained with the inpainted images, as well as augmentations of these images such as: color/saturation modifications, flipping, resizing, and rotating. 

The FINAL_UNET_MODEL.py must be trained using a dataset consisting of the impainted images and their corresponding masks. The model was trained using the GPU provided by the hackathon. The RUN_TESTS.py file will output binary masks for a set of test images using the trained model, and the CONVERT_MASK2RLE.py file converts each mask png into an RLE (run-length encoding, which was the required format for the Kaggle competition) and creates a CSV file containing the RLE data for all generated masks.