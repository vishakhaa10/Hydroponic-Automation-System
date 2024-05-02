# Hydroponic-Automation-System
# Hydroponic_Automation
Streamlit application that predicts action that has to be taken on an hydroponic system  based on stacking method(ensemble learning).

## Proposed Approach
![Untitled Diagram drawio (1)](https://user-images.githubusercontent.com/82307484/205106721-9f4c84e6-df04-46e2-bbb8-7c3fb6f8350f.png)
An IoT-based controlling system is constructed to measure the surrounding environment and the temperature, pH, TDS, and humidity of the nutrient solution.There are 4 sensors deployed in the system , three sensors—measuring pH, water temperature, and TDS—are submerged in nutrient solution inside the pipeline, and a fourth sensor—measuring environmental temperature and humidity—is installed close to the plant. The microcontroller is used to send all measurements to the cloud. Authorized users can access and evaluate the data stored on cloud servers from any location via the internet.This data is exported for analysis .
## Description
The data has been taken from recorded thingspeaks.com channel number "1013172”, the data has the following variables "pH, EC, Air temp & Humidity, Water Temp" it also records the time at "created_at". The dataset contains 100 rows of data recorded at every 5 min time interval. The hydroponic test system has 11 leafy plants and a 5-liter nutrient solution reservoir connected to it. While developing the model we noticed that the dataset was not able to be true on 25 conditions out of 28 which means if a model is trained on this data then the output would be only within the remaining 3 conditions that are true (Overfitting).While speaking with the domain expert we also heard that all 28 conditions won’t happen on a real system ,there are chances that few conditions wont happen for lifetime of the product hence we augmented data in such a way that it creates data for every conditions to be satisfied .

## Stacking Model
![Model Diagram](https://user-images.githubusercontent.com/82307484/205106663-87c0e306-b91d-48d8-bfe1-c6dfaf040ea8.png)

## Usage
install dependencies from requirement.txt

```python
pip install -r requirements.txt
```
## Output of the model in a streamlit application
<img width="212" alt="Picture1" src="https://user-images.githubusercontent.com/82307484/205107795-0aac4879-df1b-49dd-8e51-40a5dbaa40b3.png">
<img width="207" alt="Picture2" src="https://user-images.githubusercontent.com/82307484/205107814-6b499e58-e17f-40cb-ba33-684e2327848d.png">

## Contributing

Saptharishee M 
SCOPE
VIT Chennai
Chennai India
saptharisheemuthu@gmail.com

 
Vishakhaa S
SCOPE
VIT Chennai
Chennai India
vishakhaasridhar@gmail.com

 

