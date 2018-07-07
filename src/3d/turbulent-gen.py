"""
    Script for generating an artificially turbulent 3d space.
"""
import random

def surface_generation(space): #turbulent surface
    surface = []
    #rondom surface
    for i in range(1,space+1):
        for j in range(1,space+1):
            rd = random.randint(1,50)
            surface.append([i,j,rd])

    # return surface

    #surface change parameter
    surface_change = 20
    surface_mono = 10
    #sort f1
    surface_sort_f1 = []
    count = surface_change
    while (count<=len(surface)):
        tmp = surface[count-surface_change:count]
        # print(tmp)
        #decrease
        temp_sort = sorted([x[2] for x in tmp[:surface_mono]], reverse=True)
        # print(tmp[:surface_mono])
        # print(tmp[surface_mono:])
        for i in range(len(tmp[:surface_mono])):
            surface_sort_f1.append([tmp[:surface_mono][i][0],tmp[:surface_mono][i][1],temp_sort[i]])

        #increase
        temp_sort = sorted([x[2] for x in tmp[surface_mono:]])
        # print(temp_sort)
        for i in range(len(tmp[surface_mono:])):
            surface_sort_f1.append([tmp[surface_mono:][i][0], tmp[surface_mono:][i][1], temp_sort[i]])
        count = count + surface_change

    surface = [(x[1],x[0],x[2]) for x in surface_sort_f1]
    surface.sort(key=lambda x: x[0], reverse=False)

    #sort f2
    surface_sort_f2 = []
    count = surface_change
    while (count<=len(surface)):
        tmp = surface[count-surface_change:count]
        #decrease
        temp_sort = sorted([x[2] for x in tmp[:surface_mono]], reverse=True)
        # print(temp_sort)
        for i in range(len(tmp[:surface_mono])):
            surface_sort_f2.append([tmp[:surface_mono][i][0],tmp[:surface_mono][i][1],temp_sort[i]])

        #increase
        temp_sort = sorted([x[2] for x in tmp[surface_mono:]])
        # print(temp_sort)
        for i in range(len(tmp[surface_mono:])):
            surface_sort_f2.append([tmp[surface_mono:][i][0], tmp[surface_mono:][i][1], temp_sort[i]])
        count = count + surface_change


    return [(x[1],x[0],x[2]) for x in surface_sort_f2]

if __name__ == "__main__":
    space = 40
    for row in surface_generation(space):
        print (str(row[0])+"\t"+str(row[1])+"\t"+str(row[2]))

