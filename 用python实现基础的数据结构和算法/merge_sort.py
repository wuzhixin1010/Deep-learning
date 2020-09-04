def merge_sort(alist):
    n=len(alist)
    if n<=1:
        return alist#此次返回的alist是最后一次归并时作为变量的列表，也就是单个数值。return后赋值给r_list
    mid=n//2
    l_alist=merge_sort(alist[:mid])
    r_alist=merge_sort(alist[mid:])


    i=j=0
    blist=[]
    while i<len(r_alist) and j<len(l_alist):
        if r_alist[i]<=l_alist[j]:
            blist.append(r_alist[i])
            i+=1
        else:
            blist.append(l_alist[j])
            j+=1
    blist+=l_alist[j:]
    blist+=r_alist[i:]
    return blist
if __name__=="__main__":
    a=[13,2,3,1,4,6]
    print (merge_sort(a))
