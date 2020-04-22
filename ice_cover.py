import random
from random import randint
def get_dataset():
    dataset=list()
    days=list()
    days.append([118,151,121,96,110,117,132,104,125,118,125,123,110,127,131,99,126,144,136,126,91])
    days.append([130,62,112,99,161,78,124,119,124,128,131,113,88,75,111,97,112,101,101,91,110,100,130,111])
    days.append([107,105,89,126,108,97,94,83,106,98,101,108,99,88,115,102,116,115,82,110,81,96,125,104,105,124])
    days.append([103,106,96,107,98,65,115,91,94,101,121,105,97,105,96,82,116,114,92,98,101,104,96,109])
    days.append([122,114,81,85,92,114,111,95,126,105,108,117,112,113,120,65,98,91,108,113,110,105,97,105,107,88])
    days.append([115,123,118,99,93,96,54,111,85,107,89,87,97,93,88,99,108,94,74,119,102,47,82,53])
    days.append([115,21,89,80,101,95,66,106,97,87])
    days.append([109,57,87,117,91,62,65,94,86,70])
    list_of_days=list()
    for list_days in days:
        for i in list_days:
            list_of_days.append(i)
    for index in range(1855,2020):
        dataset.append([index,list_of_days[index-1855]])

    return dataset

def print_stats(dataset):
    sum=0    
    for i in range(len(dataset)):
        sum=sum+dataset[i][1]
    sum2=0
    mean_data=sum/len(dataset)
    for i in range(len(dataset)):
        sum2=sum2+((dataset[i][1]-mean_data)**2)
    deviation=sum2/(len(dataset)-1)
    deviation=deviation**(0.5)
    print(len(dataset))
    print('{:.2f}'.format(round(mean_data,2)))
    print('{:.2f}'.format(round(deviation,2)))


def regression(beta_0,beta_1):
    dataset=get_dataset()
    total_sum=0
    for i in range(len(dataset)):
        sum=pow((beta_0+(beta_1*(dataset[i][0]))-dataset[i][1]),2)
        total_sum=total_sum+sum
    total_sum=total_sum/len(dataset)

    return (round(total_sum,2))

def gradient_descent(beta_0,beta_1):
    dataset=get_dataset()
    total_sum1=0
    total_sum2=0
    for i in range(len(dataset)):
        sum1=beta_0+(beta_1*(dataset[i][0]))-dataset[i][1]
        sum2=(beta_0+(beta_1*(dataset[i][0]))-dataset[i][1])*(dataset[i][0])
        total_sum1=total_sum1+sum1
        total_sum2=total_sum2+sum2
    total_sum1=(total_sum1*2)/(len(dataset))
    total_sum2=(total_sum2*2)/(len(dataset))
    tuple1=((round(total_sum1,2)),(round(total_sum2,2)))
    return tuple1

def iterate_gradient(T,eta):
    beta_0=0
    beta_1=0
    for iteration in range(T):
        descent=gradient_descent(beta_0,beta_1)
        beta_0=beta_0-(eta*(descent[0]))
        beta_1=beta_1-(eta*(descent[1]))
        mean_squared_error=regression(beta_0,beta_1)
        print((iteration+1),"  ",(round(beta_0,2)),"  ",(round(beta_1,2)),"  ",(round(mean_squared_error,2)))

def compute_betas():
    dataset=get_dataset()
    mean_x=0
    mean_y=0
    n=len(dataset)
    for i in range(n):
        mean_x=mean_x+dataset[i][0]
        mean_y=mean_y+dataset[i][1]
    mean_x=mean_x/n
    mean_y=mean_y/n
    beta_0=0
    beta_1=0
    numerator=0
    denominator=0
    for i in range(n):
        numerator=numerator+((dataset[i][0]-mean_x)*(dataset[i][1]-mean_y))
        denominator=denominator+(pow(dataset[i][0]-mean_x,2))
    beta_1=numerator/denominator
    beta_0=mean_y-(beta_1*mean_x)
    mean_squared_error=regression(beta_0,beta_1)
    tuple_data=(beta_0,beta_1,mean_squared_error)
    return tuple_data


def predict(year):
    tuple_data=compute_betas()
    prediction=tuple_data[0]+(tuple_data[1]*year)
    return (round(prediction,2))


def gradient_descent_normalised(beta_0,beta_1,dataset):
    total_sum1=0
    total_sum2=0
    for i in range(len(dataset)):
        sum1=beta_0+(beta_1*(dataset[i][0]))-dataset[i][1]
        sum2=(beta_0+(beta_1*(dataset[i][0]))-dataset[i][1])*(dataset[i][0])
        total_sum1=total_sum1+sum1
        total_sum2=total_sum2+sum2
    total_sum1=(total_sum1*2)/(len(dataset))
    total_sum2=(total_sum2*2)/(len(dataset))
    tuple1=(total_sum1,total_sum2)
    return tuple1

def regression_normalised(beta_0,beta_1,dataset):
    total_sum=0
    for i in range(len(dataset)):
        sum=pow((beta_0+(beta_1*(dataset[i][0]))-dataset[i][1]),2)
        total_sum=total_sum+sum
    total_sum=total_sum/len(dataset)
    return total_sum

def iterate_normalized(T,eta):
    dataset=get_dataset()
    mean_x=0
    n=len(dataset)
    for i in range(n):
        mean_x=mean_x+dataset[i][0]
    mean_x=mean_x/n
    sum2=0
    for i in range(len(dataset)):
        sum2=sum2+(pow(dataset[i][0]-mean_x,2))
    deviation=sum2/(len(dataset)-1)
    deviation=pow(deviation,0.5)
    for i in range(n):
        dataset[i][0]=(dataset[i][0]-mean_x)/deviation
    beta_0=0
    beta_1=0
    for iteration in range(T):
        descent=gradient_descent_normalised(beta_0,beta_1,dataset)
        beta_0=beta_0-(eta*(descent[0]))
        beta_1=beta_1-(eta*(descent[1]))
        mean_squared_error=regression_normalised(beta_0,beta_1,dataset)
        print((iteration+1),"  ",(round(beta_0,2)),"  ",(round(beta_1,2)),"  ",'{:.2f}'.format((round(mean_squared_error,2))))

def gradient_descent_sgd(beta_0,beta_1,x,y):
    descent_beta_0=2*(beta_0+(beta_1*x)-y)
    descent_beta_1=2*(beta_0+(beta_1*x)-y)*x
    list_data=list()
    list_data.append(descent_beta_0)
    list_data.append(descent_beta_1)
    return list_data

def sgd(T,eta):
    dataset=get_dataset()
    mean_x=0
    n=len(dataset)
    for i in range(n):
        mean_x=mean_x+dataset[i][0]
    mean_x=mean_x/n
    sum2=0
    for i in range(len(dataset)):
        sum2=sum2+(pow(dataset[i][0]-mean_x,2))
    deviation=sum2/(len(dataset)-1)
    deviation=pow(deviation,0.5)
    for i in range(n):
        dataset[i][0]=(dataset[i][0]-mean_x)/deviation
    for iteration in range(T):
        beta_0=0
        beta_1=0
        index=randint(0, len(dataset))
        x=dataset[index][0]
        y=dataset[index][1]        
        data_list=gradient_descent_sgd(beta_0,beta_1,x,y)
        beta_0=beta_0-(eta*(data_list[0]))
        beta_1=beta_1-(eta*(data_list[1]))
        mean_squared_error=regression_normalised(beta_0,beta_1,dataset)
        print((iteration+1),"  ",(round(beta_0,2)),"  ",(round(beta_1,2)),"  ",(round(mean_squared_error,2)))
    

