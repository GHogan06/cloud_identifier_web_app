# cloud_identifier_web_app
A web application that allows users to upload photos of clouds in order to identify and classify them using a machine learning model.
Above can be found the files for this project. The model was trained using the script cloud_identifier_model_v2.py and the trained model itself
is included too as the file cloud_identifier_mobilenet.keras. To use this program, open cloud_web_app.py in a pycharm project and run the script as 
a server. The comment at the top of the script gives instructions on how to do this. Next, run the html page in pycharm, VSCode or any other similar program and 
try uploading the test images included to the model to yield a prediction.

Keep in mind, the model itself was trained with about 100 images per category, so it isn't always accurate. Another reason for this is that clouds in general tend to
defy classification, so one person may think a cloud a cumulonimbus cloud, while another may feel its a large cumulus cloud. Therefore, the model works best when 
the images it receives are classic examples of the various cloud types.
