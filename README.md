# DDOS-ML-Detection
This project uses a simple feedforward network built in keras to determine if incoming network packets are from one of four types of ddos attacks or are a normal request. 

# Dataset Download:
You can download the dataset used in this project here:
[Dataset link](https://drive.google.com/drive/folders/1BqMKyKUgc1U6hpfzsAAKODyG_6bAChU_?usp=sharing)

# Model Structure: 
![model](https://raw.githubusercontent.com/jacobvs/DDOS-ML-Detection/master/model.png)

# Data attributes:
@attribute SRC_ADD numeric
@attribute DES_ADD numeric
@attribute PKT_ID numeric
@attribute FROM_NODE numeric
@attribute TO_NODE numeric
@attribute PKT_TYPE {tcp,ack,cbr,ping}
@attribute PKT_SIZE numeric
@attribute FLAGS {-------,---A---}
@attribute FID numeric
@attribute SEQ_NUMBER numeric
@attribute NUMBER_OF_PKT numeric
@attribute NUMBER_OF_BYTE numeric
@attribute NODE_NAME_FROM {Switch1,Router,server1,router,clien-4,client-2,Switch2,client-5,clien-9,clien-2,clien-1,clien-14,clien-5,clien-11,clien-13,clien-0,switch1,client-4,clienthttp,clien-7,clien-19,client-14,clien-12,clien-8,clien-15,webserverlistin,client-18,client-1,switch2,clien-6,client-10,client-7,webcache,clien-10,client-15,clien-3,client-17,client-16,clien-17,clien-18,client-12,client-8,client-0,clien-16,client-13,client-11,client-6,client-3,client-9,client-19,http_client}
@attribute NODE_NAME_TO {Router,server1,Switch2,Switch1,clien-1,clien-5,clien-7,switch1,clien-11,clien-15,clien-13,clien-3,clien-9,clien-6,router,clien-4,clien-14,switch2,clien-8,clienthttp,webcache,clien-10,clien-12,webserverlistin,clien-0,clien-2,http_client,client-13,client-9,client-1,client-19,client-4,client-17,client-7,client-3,client-12,client-2,clien-18,client-16,clien-17,client-0,clien-16,client-18,client-5,client-11,client-14,client-8,client-6,client-10,clien-19,client-15}
@attribute PKT_IN numeric
@attribute PKT_OUT numeric
@attribute PKT_R numeric
@attribute PKT_DELAY_NODE numeric
@attribute PKT_RATE numeric
@attribute BYTE_RATE numeric
@attribute PKT_AVG_SIZE numeric
@attribute UTILIZATION numeric
@attribute PKT_DELAY numeric
@attribute PKT_SEND_TIME numeric
@attribute PKT_RESEVED_TIME numeric
@attribute FIRST_PKT_SENT numeric
@attribute LAST_PKT_RESEVED numeric
@attribute PKT_CLASS {Normal,UDP-Flood,Smurf,SIDDOS,HTTP-FLOOD}

# Example lines of data:
36.850655,36.85372,36.863775,0.003065,1016.449678,703384,692,0.060108,0.037172,36.84,36.877172,1,9.960601,UDP-Flood
24.11,11,573923,24,23,ack,55,-------,12,15981,16103,885665,server1,Router,49.649341,49.649341,49.659344,0,328.522947,18068.8,55,0.008446,0,49.649341,49.67935,1.030019,50.046382,Normal

# Model accuracy:
Training loss & accuracy: [1.6756146636651195, 0.8960413987239542]
![model_accuracy](https://raw.githubusercontent.com/jacobvs/DDOS-ML-Detection/master/model_accuracy.jpg)
