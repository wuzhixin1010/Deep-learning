def insert_sort(alist):
    n=len(alist)
    for i in range(1,n):
        while alist[i]<alist[i-1]:
            alist[i],alist[i-1]=alist[i-1],alist[i]
            i-=1

if __name__=="__main__":
    a=[1,5,7,6,2]
    insert_sort(a)
    print(a)