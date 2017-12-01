from __future__ import division
import csv
import random
import numpy as np
from getsamefile import getsamestockname as gstn
import tensorflow as tf

#parameters
num_labelmethod_dic={1:'SumLabel',2:'AreaLabel'}
price_location =2    #look for the csv data you uesd, find the row number of the price data.
label_method =2      #select one number from labelmethod we have.
datas_model= 1       #select a model of you stockdatas 1 means 5 mins datas, 2 means 1 mins datas
alpha =0.2           #the threshold to decide the label,by experience,0.2 is a good threshold.
T = 1                #how many day's price datas  we care to generate the labels.

#dir parameters
generate_aim='train'  #aim to generate the data
csvpath='/media/lix/51d7efd5-72f0-450c-ab09-f65259f11e25/stockdata'   #the dir of your stockdata
savedir = '/media/lix/51d7efd5-72f0-450c-ab09-f65259f11e25/dataforc'  #the dir to save processed matrix datas
savepath=savedir+'/'+generate_aim+'/'+num_labelmethod_dic[label_method]    #detil file name of the matrix datas


# load the data from aim file and put the price data in pricelist
def loadfile(filename,price_location):
    csv_reader = csv.reader(open(filename))
    print(type(csv_reader))
    pricelist = []
    for i in csv_reader:
        pricelist.append(i[price_location])
    return pricelist

#label method 1
def GetLabelWithArea(Tday,alpha):
    uparea = 0
    downarea = 0

    for i in range(len(Tday)):
        if i == 0:
            pass
        else:
            delta1=float(Tday[i])-float(Tday[0])
            delta=float(Tday[i-1])-float(Tday[0])
            if (delta1*delta) >= 0:
                area = (delta1+delta)*0.5
                if area >= 0:
                    uparea += area
                else:
                    downarea += area
            else:
                area1 = 0.5*abs(delta1)*abs(delta1)/(abs(delta)+abs(delta1))
                if delta1 >= 0:
                    uparea += area1
                else:
                    downarea += area1
                area2 = 0.5*abs(delta)*abs(delta)/(abs(delta)+abs(delta1))
                if delta >= 0:
                    uparea += area2
                else:
                    downarea += area2
    if downarea == 0:
        y = [0, 1]
    else:
        if abs(uparea/downarea) <= alpha:
            y = [1,0]
        else:
            y = [0,1]
    return y



# label method 2
def GetLabelWithSum(Tday, alpha):
    upsum = 0
    downsum = 0

    for i in range(len(Tday)):
        if (float(Tday[i]) - float(Tday[0])) >= 0:
            upsum += (float(Tday[i]) - float(Tday[0]))
        else:
            downsum += (float(Tday[i]) - float(Tday[0]))

    if downsum == 0:
        y = [0, 1]
    else:
        if abs(upsum / downsum) <= alpha:
            y = [1, 0]
        else:
            y = [0, 1]
    return y


def five_mins_dataprocessing(pricelist):
    length = len(pricelist)
    days = int(length / 48)
    final_matrix_all = []
    time_list = []
    # day_list = []
    label_list = []

    for day in range(8, days + 1 - T):
        # day_list.append(day)
        t = random.randint(0, 47)
        Tday = pricelist[(day - 1) * 48 + t:(day + T - 1) * 48 + t + 1]
        label = GetLabelWithArea(Tday, alpha)
        label_list.append(label)
        firstslice = pricelist[(day - 8) * 48 + t + 1:(day - 1) * 48 + t + 1]
        # matrix2 = np.zeros([7,6])
        # for index in range(7):
        # matrix2[index,:] = firstslice[(index+1)*48+t-6:(index+1)*48+t]
        # secondslice = firstslice[t+48:48*8+t]
        matrix1 = np.array(firstslice).reshape(7, 48)
        matrix = np.zeros([7, 48])
        matrix[:, 0:] = matrix1
        normal_matrix = matrix / float(firstslice[335])
        addlist1 = [i for i in range(t + 1)]
        addlist2 = [i for i in range(t + 1, 48)]
        addlist2.extend(addlist1)
        addlist = [i / 48 for i in addlist2]
        addmatrix = np.array(addlist)
        for i in range(6):
            addmatrix = np.row_stack((addmatrix, addlist))
        addmatrix1 = np.zeros([7, 48])
        addmatrix1[0][0] = 1
        final_matrix = np.concatenate((normal_matrix, addmatrix, addmatrix1))
        final_matrix = final_matrix.reshape(3, 7, 48)
        # addlist = [0 for i in range(54)]
        # addlist[53-t] = 1
        # final_matrix = np.row_stack((normal_matrix,addlist))
        final_matrix_all.append(final_matrix)
        time_list.append(t)
    return final_matrix_all, label_list, time_list


# saved data is a string not a matrix. we have to change the data type before training

if __name__ == "__main__":

    samefilename, filepath = gstn(csvpath)
    address = [csvpath + '/' + file for file in filepath]
    # filename_list = []
    for index1 in range(len(address)):
        for index2 in range(len(samefilename)):
            filename = address[index1] + '/' + samefilename[index2]

            year = filename[58:62]
            # filename_list.append(filename)
            pricelist = loadfile(filename,price_location)
            if datas_model == 1:
                final_matrix, label_list, time_list = five_mins_dataprocessing(pricelist)
            else:
                pass
                #final_matrix, label_list, time_list = one_mins_dataprocessing(pricelist)
            processedname = samefilename[index2] + year
            # datas = {'final_matrix':final_matrix,'label_list':label_list,'time_list':time_list}
            name = (savepath + '/' + processedname)
            writer = tf.python_io.TFRecordWriter(name)

            for j in range(len(final_matrix)):
                data_raw = np.array(final_matrix[j]).tostring()
                label_raw = label_list[j]

                example = tf.train.Example(features=tf.train.Features(feature={
                    'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw])),
                    'lable_raw[0]': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_raw[0]])),
                    'lable_raw[1]': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_raw[1]]))}))
                writer.write(example.SerializeToString())
            writer.close()
