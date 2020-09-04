def shell_sort(alist):
    n=len(alist)
    gap=n//2
    while gap>0:
        for j in range(gap,n):
            i=j
            while i>0:
                if alist[i]<alist[i-gap]:
                    alist[i],alist[i-gap]=alist[i-gap],alist[i]
                    i-=gap
                else:
                    break

        gap=gap//2
    return alist
if __name__=="__main__":
    a=[1,2,4,4,3,2,5,6,4,6,3,4,5,6,7,8,3,2,4,3,30,2,1,4,33,99,5,87,85,7]
    shell_sort(a)
    print(a)