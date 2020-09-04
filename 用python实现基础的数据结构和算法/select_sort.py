def select_sort(alist):
    n=len(alist)
    for i in range(n-1):
        max=n-1-i
        for j in range(n-1-i):
            if alist[max]<alist[j]:
                max=j
        alist[n-1-i],alist[max]=alist[max],alist[n-1-i]
    return alist
if __name__=="__main__":
    a=[1,3,4,2,5]
    select_sort(a)
    print(a)



