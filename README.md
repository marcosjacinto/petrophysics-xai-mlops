# Sonic Log Synthetization with Machine Learning, Explainable Artificial Intellinge and MLOps

This project was developed by [Marcos Jacinto](https://www.linkedin.com/in/marcos-jacinto/).

About this project
__________

Compressional and shear sonic traveltime logs (DTC and DTS, respectively) are crucial for subsurface characterization and seismic-well tie. However, these two logs are often missing or incomplete in many oil and gas wells. Therefore, many petrophysical and geophysical workflows include sonic log synthetization or pseudolog generation based on multivariate regression or rock physics relations.

From March 1, 2020 to May 7, 2020, the SPWLA hosted a contest aiming to predict the DTC and DTS logs from seven conventional logs using machine-learning methods.
Although the competitors got excellent results, their code and the requirements of the competition still showed a disconnect between data-driven methodologies in Geoscience and software engineering and development skills required to create real solutions.

Therefore, the objective of this project is to create a DTC prediction model with Machine Learning using the same data provided by the SPWLA. MLOps and Explainable Artificial Intelligence techniques were included as part of the methdology. This was done in order to exemplify what is required from a data-driven solution in a software development environment 


*xAI*: XAI proposes the creation of a set of methodologies capable of producing explanations and promoting the interpretation of complex models such as deep neural networks while maintaining a high level of performance.

*MLOps*: The term MLOps is defined as “the extension of the DevOps methodology to include Machine Learning and Data Science assets as first-class citizens within the DevOps ecology” Source: [MLOps SIG](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2020/MLOpsRoadmap2020.md).


The Data
__________
The data used by this project was provided by the SPWLA in its Machine Learning Contest (2020). The data contains the following logs: caliper, neutron porosity, gamma ray (GR), deep resistivity, medium resistivity, photoelectric factor, and bulk density, as well as two sonic logs (DTC and DTS) as the target. 


How to
___________

Create a new environment. Then, use the following command in the terminal to install the requirements:

`pip install -r requirements.txt`

If you have acess to the repository that contains the data you can use the following command to retrieve it:

`dvc pull`

If you don't have acess to the repository you can find the data in the GitHub [repository provided by SPWLA](https://github.com/pddasig/Machine-Learning-Competition-2020).

To process the raw data, i. e., clean outliers and null values, as well as apply yeo-johnson's power transformation use the following command:

`python data/process_data.py`

After processing the data you can train a new neural network with:

`python model/train_neural_network.py`

To see the details of each training/experiment and compare them use MLFlow:

`mlflow ui`

You must use these commands from the root directory of the project.