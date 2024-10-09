## User guide for the NLP sentiment analysis challenge

- Run all the cells in the jupyter notebook `Aufgabe_A.ipynb`. This will preprocess the data, train and save the model. At the end the model is evaluated on the validation set.
- Now build the docker image using ```bash docker build -t model-app .```
- Run the container using ```bash docker run -p 5000:5000 -e BATCH_SIZE=16 model-app```. The BATCH_SIZE is an optional environment variable that can be set by the user depending on the hosting device. It defaults to 32.


<br>

This app has two endpoints: <br>
**The predict endpoint**  is useful for single predictions. You give a single string as input and you expect a single sentiment prediction for that string. Send a post request to http://localhost:5000/predict with json body: `{"text": YOUR_INPUT_TEXT_STRING}` <br>
**The batch_predict endpoint** is useful for generating predictions for a list of input texts. The predictions are generated in batches of size  BATCH_SIZE internally, and then concatenated together to return a list of predicted labels of the same size of the list of input texts. Send a post request to http://localhost:5000/batch_predict with json body: `{"texts": YOUR_LIST_OF_INPUT_TEXT_STRINGS}` <br>