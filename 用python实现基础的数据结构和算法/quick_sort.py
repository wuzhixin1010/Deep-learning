def quick_sort(alist,first,last):
    if first>last:
        return
##################非常重要！！！！不然递归会出错！！！！！
    a=alist[first]
    i=last
    j=first
    while i>j:
        while i>j and alist[i]>=a:
            i-=1
        alist[j] = alist[i]
        while i>j and alist[j]<a:
            j+=1
        alist[i]=alist[j]
    alist[i]=a

    quick_sort(alist,first,i-1)
    quick_sort(alist,i+1,last)
if __name__=="__main__":
    b=[3,5,1,0,3,4,2,5,6]
    quick_sort(b,0,len(b)-1)
    print(b)
