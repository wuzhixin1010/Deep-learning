class Node(object):
     def __init__(self,elem):
         self.elem=elem
         self.next=None
class single_link_list(object):
    def __init__(self,node=None):
        self.head=node
    def is_empty(self):
        return self.head==None
    def length(self):
        cur=self.head
        count=0
        while cur != None:
            count+=1
            cur=cur.next
        return count
    def travel(self):
        cur=self.head
        while cur!=None:
            print(cur.elem,end=" ")
            cur=cur.next
    def append(self,item):
        node=Node(item)
        if self.is_empty():
            self.head=node
        else:
            cur=self.head
            while cur.next!=None:
                cur=cur.next
            cur.next=node
    def add(self, item):
        cur=self.head
        node=Node(item)
        if cur==None:
            self.head=node
        else:
            node.next=cur
            self.head=node
    def insert(self,pos,item):
        node=Node(item)
        if pos<=0:
            self.add(item)
        elif pos>(self.length()-1):
            self.append(item)
        else:
            pre=self.head
            count=0
            while count<(pos-1):
                pre=pre.next
                count+=1
            node.next=pre.next
            pre.next=node
    def remove(self,item):
        cur=self.head
        pre=None
        while cur!=None:
            if cur.elem==item:
                if self.head==item:
                    self.head=self.next
                else:
                    pre.next=cur.next
                break
            else:
                pre=cur
                cur=cur.next




if __name__ == "__main__":
    ll=single_link_list()
    print(ll.is_empty())
    print(ll.length())
    ll.append(1)
    print(ll.is_empty())
    print(ll.length())
    ll.append(2)
    print(ll.is_empty())
    print(ll.length())
    ll.add(3)
    print(ll.is_empty())
    print(ll.length())
    ll.insert(2,3)
    ll.travel()
    ll.remove(2)
    print(ll.is_empty())
    print(ll.length())
    ll.travel()
