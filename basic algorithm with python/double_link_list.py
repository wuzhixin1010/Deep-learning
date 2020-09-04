class Node(object):
    """结点"""
    def __init__(self, item):
        self.elem = item
        self.next = None
        self.prev = None


class DoubleLinkList(object):
    """双链表"""
    def __init__(self, node=None):
        self.__head = node

    def is_empty(self):
        """链表是否为空"""
        return self.__head == None

    def length(self):
        """链表长度"""
        # cur游标，用来移动遍历节点
        cur = self.__head
        # count记录数量
        count = 0
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """遍历整个链表"""
        cur = self.__head
        while cur != None:
            print(cur.elem, end=" ")
            cur = cur.next
        print("")

    def add(self, item):
        """链表头部添加元素，头插法"""
        node = Node(item)
        node.next = self.__head
        self.__head = node
        node.next.prev = node

    def append(self, item):
        """链表尾部添加元素, 尾插法"""
        node = Node(item)
        if self.is_empty():
            self.__head = node
        else:
            cur = self.__head
            while cur.next != None:
                cur = cur.next
            cur.next = node
            node.prev = cur
    def insert(self,pos,item):
        cur=self.__head
        count=0
        if pos<=0:
            self.add(item)

        elif pos>(self.length()-1):
            self.append(item)
        else:
            while count<(pos-1):
                count+=1
                cur=cur.next
            node=Node(item)
            node.next=cur.next
            node.pre=cur
            cur.next.pre=node
            cur.next=node
    def remove(self,item):
        cur=self.__head
        if cur!=None:
            while cur.elem==item:
                if cur==self.__head:
                   self.__head=cur.next
                   cur.next.pre=None
                elif cur.next==None:
                   cur.pre.next=None
                   cur.pre=None
                else:
                    cur.next.pre=cur.pre
                    cur.pre.next=cur.next
                break
            cur=cur.next
    def search(self,item):
         cur=self.__head
         while cur!=None:
             if cur.elem==item:
                 return Ture
             else:
                 cur=cur.next
         return False

if __name__ == "__main__":
    ll = DoubleLinkList()
    print(ll.is_empty())
    print(ll.length())

    ll.append(1)
    print(ll.is_empty())
    print(ll.length())

    ll.append(2)
    ll.add(8)
    ll.append(3)
    ll.append(4)
    ll.append(5)
    ll.append(6)
    ll.travel()
    # 8 1 2 3 4 5 6
    ll.insert(-1, 9)  # 9 8 1 23456
    ll.travel()
    ll.insert(3, 100)  # 9 8 1 100 2 3456
    ll.travel()
    ll.insert(10, 200)  # 9 8 1 100 23456 200
    ll.travel()
    ll.remove(100)
    ll.travel()
    ll.remove(9)
    ll.travel()
    ll.remove(200)
    ll.travel()