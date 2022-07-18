# Make your own submission

In order to participate in the Shift Challenge 2.0 you have to submit your algorithm wrapped in a docker file.

### Create your own docker

In order to create your algorithm docker you have to write your own code in process.py.
More specifically you have to inherit from **_ShippingBaseAlgorithm_** and implement the **_load_model_** and 
**_predict_** template methods. You can also do anything you want in the __init__ of your class. Do not change the 
source code of **_ShippingBaseAlgorithm_** as you may face some issues in the submission phase. The template docker 
works with pandas DataFrame as is very convenient regarding tabular data tasks.

* **load_model**: In this method you have to define the way that your model is loaded. This generally depends on 
 the library you use to train your model and the way you serialise the model in the training phase. You are free
 to use any python library for your model (include the libraries in the ./requirements.txt). All the dependencies, in 
 terms of source code and files, should be stored in the ./model folder. In our example docker tutorial we load all 10 
 member of our ensemble together with the required file about the scaling.

* **_predict_**: Write your code in order to make the inference of the model. In generally the output should be a pandas
DataFrame containing 3 columns ("time_id", "power", "uncertainty"). You can use the self.form_outputs to ensure that 
your data is on the correct format.


### Tips: 
The docker take into account specific folders and files as described in the Dockerfiles. So you can add and
modify the following folder or files.
  * process.py (write your inference process)
  * /utils (you can add as many pythons files as you want)
  * /model (you can add both models statistics or python scripts)

In case that the temple does not meet the requirements of your method you can create an algorithm docker from scratch as 
described in the documentation of grand-challenge. We are strongly advised you to change the current table to 
create the docker

The only requirements in the input and the outpath as well as the output format.
You should read your input from /input/merchant-vessel-features.json and write your output in 
/output/vessel-power-estimates.json. You check the output format in the /test/expected_output.json .

### Development

For the development phase of the algorithm docker you can Modify the following paths on the /utils/eval_utils.py 
to match your local path.

DEFAULT_OUTPUT_PATH = Path("/output")
DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_MODEL_PATH = Path("/opt/algorithm/model")

### Test you algorithm docker

You can test your algorithm by running the ./test.sh file.
The sample of test input data is stored in the ./test folder.

If you face problem  with you submission while your ./test.sh work fine is probably due to memory error.
To extensively validate your submissions, create a fake test dataset of 2.600.000 points and store it in the /test folder.

### Who do I talk to?

If you have any question don't hesitate to concat us. 

e.tsompopoulou@deepsea.ai  
a.athanasopoulos@deepsea.ai  
a.nikitakis@deepsea.ai
