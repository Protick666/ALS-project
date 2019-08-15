
import ijson
import numpy
import numpy.matlib
from scipy.sparse import csc_matrix,lil_matrix
from scipy import sparse
from operator import itemgetter


def sp_calc_error(temp, height,width,k):
    tot_err=0.0
    global main_mat
    tot=0
    for p in range(0,len(main_mat.nonzero()[0])):
        i,j = main_mat.nonzero()[0][p], main_mat.nonzero()[1][p]
        tot_err = tot_err + (main_mat[i, j]-temp[i, j]) * (main_mat[i, j]-temp[i, j])
        tot = tot + 1

    return (tot_err/tot) ** (.5)


def calc_error(temp, height,width,k):
    tot_err=0
    global main_mat
    for p in range(0,len(main_mat.nonzero()[0])):
        i,j = main_mat.nonzero()[0][p], main_mat.nonzero()[1][p]
        tot_err = tot_err + (main_mat[i,j]-temp[i,j]) * (main_mat[i,j]-temp[i,j])

    #return tot_err
    '''
    sum1=0
    for row in range(0,height):
        temp_row=0
        for col in range(0,width):
            temp_row = temp_row + main_mat[row, col]
        temp_row = temp_row * temp_row
        sum1 = sum1 + lu * temp_row

    sum2 = 0
    for col in range(0, width):
        temp_row = 0
        for row in range(0, height):
            temp_row = temp_row + main_mat[row, col]
        temp_row = temp_row * temp_row
        sum2 = sum2 + lv * temp_row
    '''

    return  tot_err




def update_U(height, width, k):
    global U
    global V
    global main_mat
    for row in range (1,height):
        #print("column", row)
        fill_col=[]
        tr=main_mat[row, :]
        for pp in tr.nonzero()[1]:
            fill_col.append(pp)

        store = numpy.matlib.zeros((k, k))
        for col in fill_col:
            store=V[:,col] * V[:,col].transpose() + store

        id = numpy.identity(k);
        store = store + .1 * id
        store = store.I

        sec_store=numpy.matlib.zeros((k, 1))

        for col in fill_col:
            #print("seccolumn", col)
            sec_store = sec_store + main_mat[row, col] * V[:, col]

        fin = store * sec_store
        fin = fin.transpose()
        U[row, :] = fin


def update_V(height, width, k):
    global U
    global V
    global main_mat

    for col in range (0,width):

        fill_row=[]
        tmp=main_mat[:,col]
        for t in tmp.nonzero()[0]:
            fill_row.append(t)

        store = numpy.matlib.zeros((k, k))
        for row in fill_row:
            store = U[row, :].transpose() * U[row, :] + store

        id = numpy.identity(k);
        store = store + .1 * id
        store = store.I

        sec_store=numpy.matlib.zeros((k, 1))
        for p in fill_row:
            #print("p", p)
            j=U[row, :]
            j=j.transpose()
            sec_store = sec_store + main_mat[p, col] * j

        fin = store * sec_store

        V[:, col] = fin











f = open('train.json')
'''
objects = ijson.items(f, 'item')
for obj in objects:
    print (obj["itemID"])
'''
objects = ijson.items(f, '.item')
count = 0

alltuple = []
for l in f:
    s = l.find("'itemID': '")
    sub=l[s+11:]
    item=sub.split('\'')[0]
    s = l.find("'rating': ")
    sub = l[s + 10:]
    rating = float(sub.split(',')[0])
    s = l.find("'reviewerID': '")
    sub = l[s + 15:]
    user = sub.split('\'')[0]
    tup = (user,item,rating)
    alltuple.append(tup)
    count=count+1
    if count == 200000:
        break

tot_ex = 30000

srt = []
how_vis={}
user_all = []
data = {}

height_main=0
for tup in alltuple:
    if tup[0] not in how_vis:
        how_vis[tup[0]] = 1
        user_all.append(tup[0])
        if tup[0] not in data:
            data[tup[0]] = []
        data[tup[0]].append((tup[1], tup[2]))
    else:
        how_vis[tup[0]] = how_vis[tup[0]] + 1

        if tup[0] not in data:
            data[tup[0]] = []
        data[tup[0]].append((tup[1], tup[2]))

for i in range(0, len(user_all)):

    srt.append((user_all[i], how_vis[user_all[i]]))

srt.sort(key=lambda tup: tup[1],reverse=True)

cnt=0


t_end = 0
v_end = 0
t_user = 0
v_user = 0

take = 2000
split=int(take*.1)

alltuple = []
cnt = 0

for i in range(0, len(srt)):
    t = srt[i]
    for j in range(0, len(data[t[0]])):
        alltuple.append((t[0], data[t[0]][j][0], data[t[0]][j][1]))
        cnt = cnt + 1
    if cnt > split*6:
        t_end = cnt
        t_user = i
    if cnt > split*8:
        v_end = cnt
        v_user = i
    if cnt > take:
        take = cnt
        break






















user_vis={}
item_vis={}
width=0
height_main=0
for tup in alltuple:
    if tup[0] not in user_vis:
        user_vis[tup[0]]=height_main
        height_main=height_main+1
    if tup[1] not in item_vis:
        item_vis[tup[1]]=width
        width=width+1

print(height_main,width)

data = {k: [] for k in range(height_main)}

for tup in alltuple:
    u_id = user_vis[tup[0]]
    pro_id = item_vis[tup[1]]
    r=tup[2]
    data[u_id].append((pro_id,r))


cons=[.01]
lat=[5]

ac_parameter=(0,0,0)
ac_ref=-1

best_V = 0

for k in lat:
    for lu in cons:
        for lv in cons:
            #print("one it")
            height= t_user
            main_mat = lil_matrix((height, width), dtype=numpy.float)

            V = numpy.matlib.rand((k, width))
            U = numpy.matlib.zeros((height, k))
            for i in range(0,height):
                for j in range (0,len(data[i])):
                    iti = data[i][j][0]
                    rt = data[i][j][1]
                    #print(user_vis[tup[0]],item_vis[tup[1]])
                    main_mat[i, iti] = rt
            print("done with data store")
            pre=-1
            for it in range(1,100):
                #temp = sparse.lil_matrix(U * V)
                uu = sparse.lil_matrix(U)
                vv = sparse.lil_matrix(V)
                temp= uu * vv

                error = calc_error(temp, height, width, k)
                if pre != -1:
                    diff=abs(error-pre)
                    print("diff",diff)
                    if diff<.0001:
                        break
                pre=error
                print("inside main loop")
                update_U(height, width, k)
                update_V(height, width, k)


            print("ola")

            height = v_user

            main_mat = lil_matrix((height, width), dtype=numpy.float)

            U = numpy.matlib.zeros((height, k))

            for i in range(t_user, v_user):
                for j in range(0, len(data[i])):
                    iti = data[i][j][0]
                    rt = data[i][j][1]
                    # print(user_vis[tup[0]],item_vis[tup[1]])
                    main_mat[i - t_user, iti] = rt
            update_U(height, width, k)

            uu = sparse.lil_matrix(U)
            vv = sparse.lil_matrix(V)
            temp = uu * vv


            err = sp_calc_error(temp, height, width, k)
            print("valid error ",err)
            if ac_ref == -1:
                ac_ref = err
                ac_parameter = (lu, lv, k)
                best_V = vv.copy()
            elif ac_ref > err:
                ac_ref = err
                ac_parameter = (lu, lv, k)
                best_V = vv.copy()



print(ac_parameter)