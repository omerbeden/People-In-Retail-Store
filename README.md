# People-In-Retail-Store
People analysis in retail store

![couldn't loaded](/images-rsc/app-gif.gif)

This is a kind of project that make analysis on people in store. For now, it's providing only people count data. Aim of this project is increase sales amount by providing  useful data.
 You can see amount of people in specific field. This fields can be more likely hallway or something like that. You can also see points that people walk around in store. Started color 
 is light blue , if a person stay same position more than one frame it gets dark.
 
_Object tracking system_ can be implemented (I actually  implemented DeepSort already but I did't use it because of Raspberry Pi limitations ) to provide diversity data.
You can obtain the time that people walk around in specific field. For instance, people in field1 walk around in that field average 10 minutes. 

#Required Libraries
* Flask
* Flask-Socketio
* OpenCv
* Numpy
* Pandas
* Shapely
* xlsxwriter
* Threading
* Python-Firebase
* Tensorflow lite run time interpreter



#Structure
Firstly, the application runs in real-time. I use socketio to communicate to flask server in real-time. Basic idea updating data is sending socketio request to flask server every in 1 sec. So, this triggers the server to update data.
![couldn't loaded](/images-rsc/architecture.PNG)

#Report
Server create a excel file by getting data from Firebase when clicked report button. Excel file contains counting data and some charts according to that data.
![couldn't loaded](/images-rsc/excel ss.PNG)


