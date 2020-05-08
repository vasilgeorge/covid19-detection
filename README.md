# covid19-detection


Covid19-detection is a Machine Learning model used to detect Covid19 from patients' chest X-Rays images taken from here: https://github.com/UCSD-AI4H/COVID-CT.

The Machine Learning model used in this work is a Random Forest Classifier that classifies an X-Ray image as positive or negative indicating whether the patient has contracted Covid19 or not with a 95% detection accuracy.

The feature vector passed to the Random Forest Classifier consists of features extracted from the GLCM matrix of the chest X-Ray image after it has been denoised and has the lungs segmented from the original image.

![Image of a Covid19 Positive](https://radiologyassistant.nl/assets/2-chest-filmc.jpg)


# Disclaimer

The model presented in this work is not to be used for any medical purposes. It has been developed out of personal interest on the subject and under no circumstances do I allege that it can be deployed to accurately detect Covid19.


