import os
csvpath ='/media/lix/51d7efd5-72f0-450c-ab09-f65259f11e25/stockdata'

"""
@return:
    samestock_name:  the same stock file in different files.
    FileName: wenjianjie name

"""
def getsamestockname(path):
    FileName = os.listdir(path)
    FileName = [file for file in FileName if file.endswith('csv')]


    files = []

    for filepath in FileName:
        filename = os.listdir(path + '/' + filepath)
        files.append(filename)


    samestock = {}
    for index in range(len(files)):
        for file in files[index]:
            if file not in samestock:
                samestock[file] = 1
            else:
                samestock[file] += 1

    samestock_name = []
    for file in samestock.keys():
        if samestock[file] == 9:
            samestock_name.append(file)
        else:
            pass
    return samestock_name, FileName

if __name__ == "__main__":
	files,FileName = getsamestockname(csvpath)
	print(len(files))