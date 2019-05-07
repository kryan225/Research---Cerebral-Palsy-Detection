# Research---Cerebral-Palsy-Detection
Using OpenCV, AWS ML, and AWS Sagemaker we attempt to predict signs of cerebral palsy in premature babies


## To Run the Dense_Optical_Flow.py file:

Install Anaconda [For Windows](https://docs.anaconda.com/anaconda/install/windows/),
                 [For Linux](https://docs.anaconda.com/anaconda/install/linux/),
                 [For Mac](https://docs.anaconda.com/anaconda/install/mac-os/)

### Create a virtual environment:

1. Open an ancaconda prompt
		
2. Type `conda create -n yourenvironmentname`
		
3. Activate your new env with `source activate yourenvironmentname`

[Creating Virtual Environments with Anaconda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

*Note: You can also create a new environment using the GUI provided in the Anaconda Navigator*

### Final Steps

Install spyder using the Anaconda Navigator. I used Spyder2.3.8 but the latest version should be fine

Use spyder to open, run, or edit the Dense_Optical_Flow.py program

### Turnover stuff

Things to give Prof. Patterson

* Database access:
	* Done. Database is on AWS under Patterson's root credentials
* Ability to convert raw accelerometer in the database to generated in the db Python script that runs outside of AWS to capture the raw accelerometer data and put the generated data into AWS.
	* Done. The python script is in `2019_Summer_Workflow/01_Accelerometer_Raw_To_Generated/feature_extraction.ipynb`
* Ability to convert raw rgb frames in the database to optical flow images in the db.  A Python script that runs outside of AWS to capture the raw accelerometer data and put the generated data into AWS.
	* Done. The python script is in `2019_Summer_Workflow/02_Video_Raw_To_Generated/dense_optical_flow.py`
* Ability to convert depth frames in the database into depth-flow images
	* Not done.  First task for summer students


		
