def bubble(alist):
    n=len(alist)
    for j in range(0,n-1):
        ##共需要n-1次交换，第一次交换设为0，以便于后面减，则最后一次交换为n-2，所以要range到n-1
        count=0
        for i in range(0,n-1-j):
            #总共有n个数字，要交换n-1次，从0开始计，所以要最后一次交换为n-2，所以range到n-1，再考虑第j次
            if alist[i]>alist[i+1]:
                alist[i],alist[i+1]=alist[i+1],alist[i]
                count+=1
        if 0==count:
            return alist
    return alist
if __name__=="__main__":
    a=[5,8,7,6,3]
    print(a)
    b=bubble(a)
    print(b)