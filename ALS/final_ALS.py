
import ijson
import numpy
import numpy.matlib
from scipy.sparse import csc_matrix,lil_matrix
from scipy import sparse
from operator import itemgetter
import pandas as pd


def sp_calc_error(temp, height,width,k):
    tot_err=0.0
    global main_mat
    tot=0
    l = len(main_mat.nonzero()[0])
    tm1 = main_mat.nonzero()[0].copy()
    tm2 = main_mat.nonzero()[1].copy()
    for p in range(0,l):
        #print(p,l)
        #i,j = main_mat.nonzero()[0][p], main_mat.nonzero()[1][p]
        i, j = tm1[p], tm2[p]
        #print(main_mat[i,j]-temp[i,j])
        tot = tot + 1
        tot_err = tot_err + (main_mat[i,j]-temp[i,j]) * (main_mat[i,j]-temp[i,j])

    return (tot_err/tot) ** (.5)


def calc_error(temp, height,width,k):
    global U
    global V
    tot_err=0
    global main_mat
    l = len(main_mat.nonzero()[0])
    tm1 = main_mat.nonzero()[0].copy()
    tm2 = main_mat.nonzero()[1].copy()

    for p in range(0,l):
        #print(p,l)
        #i,j = main_mat.nonzero()[0][p], main_mat.nonzero()[1][p]
        i, j = tm1[p], tm2[p]
        #print(main_mat[i,j]-temp[i,j])
        tot_err = tot_err + (main_mat[i,j]-temp[i,j]) * (main_mat[i,j]-temp[i,j])

    #return tot_err
    sum1 = 0
    sum2 = 0


    for row in range(0,height):
        temp_row=0
        for col in range(0,k):
            temp_row = temp_row + U[row, col]*U[row, col]

        sum1 = sum1 + lu * temp_row

    
    for col in range(0, width):
        temp_row = 0
        for row in range(0, k):
            temp_row = temp_row + V[row,col]*V[row,col]

        sum2 = sum2 + lv * temp_row


    return sum1 + sum2 + tot_err




def update_U(height, width, k, lu, lv):
    global U
    global V
    global main_mat

    for row in range (0,height):
        #print("column", row)
        fill_col=[]

        tr = main_mat[row, :]
        for pp in tr.nonzero()[1]:
            fill_col.append(pp)
        #print("fill",len(fill_col))

        store = numpy.matlib.zeros((k, k))
        for col in fill_col:
            store = V[:,col] * V[:,col].transpose() + store

        id = numpy.identity(k);
        store = store + lu * id
        store = store.I

        sec_store=numpy.matlib.zeros((k, 1))

        for col in fill_col:
            #print("seccolumn", col)
            sec_store = sec_store + main_mat[row, col] * V[:, col]

        fin = store * sec_store
        fin = fin.transpose()
        U[row, :] = fin


def update_V(height, width, k, lu, lv):
    global U
    global V
    global main_mat
    for col in range (0,width):

        fill_row=[]
        tmp=main_mat[:,col]
        for t in tmp.nonzero()[0]:
            fill_row.append(t)
        #print("fill col", len(fill_row))

        store = numpy.matlib.zeros((k, k))
        for row in fill_row:
            store = U[row, :].transpose() * U[row, :] + store

        id = numpy.identity(k);
        store = store + lv * id
        store = store.I

        sec_store=numpy.matlib.zeros((k, 1))
        for p in fill_row:
            #print("p", p)
            j=U[p, :]
            j=j.transpose()
            sec_store = sec_store + main_mat[p, col] * j

        fin = store * sec_store

        V[:, col] = fin




df = pd.read_excel("ratings_train.xlsx", header=None)
train_all = df.as_matrix()



df = pd.read_excel("ratings_validate.xlsx", header=None)
val_all = df.as_matrix()








cons=[.01,.1,1,10]
lat=[10,20,40]

ac_parameter=(0,0,0)
ac_ref=-1

best_V = 0

for k in lat:
    for lu in cons:
            lv = lu
            #print("one it")
            height = len(train_all)
            width = len(train_all[0])
            height = 1000
            width = 100
            print("K ", k)
            print("LU", lu)

            main_mat = lil_matrix((height,width), dtype=numpy.float)

            V = numpy.matlib.rand((k, width))
            U = numpy.matlib.zeros((height, k))
            for i in range(0,height):
                for j in range (0,width):
                    if train_all[i][j] == -1:
                        #print("yo")
                        continue
                    main_mat[i, j] = float(train_all[i][j])
                    #print(float(train_all[i][j]))

            pre=-1
            for it in range(1,30):

                uu = lil_matrix(U)
                vv = lil_matrix(V)
                temp = uu * vv

                error = calc_error(temp, height, width, k)
                print(k,lu,lv)
                #print("local error ",error)
                if pre != -1:
                    diff=abs(error-pre)
                    #print("diff ",diff)
                    if diff <= 1:
                        break
                pre = error

                update_U(height, width, k, lu, lv)
                update_V(height, width, k, lu, lv)

            uu = sparse.lil_matrix(U)
            vv = sparse.lil_matrix(V)
            temp = uu * vv

            err = sp_calc_error(temp, height, width, k)
            print("training error", err)

            height = len(val_all)
            width = len(val_all[0])
            height = 500
            width = 100

            main_mat = lil_matrix((height, width), dtype=numpy.float)

            U = numpy.matlib.zeros((height, k))

            for i in range(0,height):
                for j in range (0,width):
                    if val_all[i][j] == -1:
                        continue
                    main_mat[i, j] = val_all[i][j]

            update_U(height, width, k, lu, lv)

            uu = sparse.lil_matrix(U)
            vv = sparse.lil_matrix(V)
            temp = uu * vv


            err = sp_calc_error(temp, height, width, k)
            print("validation error",err)

            if ac_ref == -1:
                ac_ref = err
                ac_parameter = (lu, lv, k)
                best_V = vv.copy()
            elif ac_ref > err:
                ac_ref = err
                ac_parameter = (lu, lv, k)
                best_V = vv.copy()



print(ac_parameter)
