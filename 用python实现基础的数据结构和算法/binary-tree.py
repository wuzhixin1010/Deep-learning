class Node(object):
    def __init__(self,item):
        self.elem=item
        self.lchild=None
        self.rchild=None
class tree(object):
    def __init__(self):
        self.root=None
    def add(self,item):
        node=Node(item)
        if self.root is None:
            self.root=node
            return
        queue=[self.root]
        while queue:
            curroot=queue.pop(0)
            if curroot.lchild is None:
                curroot.lchild=node
                return
            else:
                queue.append(curroot.lchild)
            if curroot.rchild is None:
                curroot.rchild=node
                return
            else:
                queue.append(curroot.rchild)
    def breadth_travel(self):
        if self.root is None:
            return
        queue=[self.root]
        while queue:
            curroot=queue.pop(0)
            print (curroot.elem,end=" ")
            if curroot.lchild is not None:
                queue.append(curroot.lchild)
            if curroot.rchild is not None:
                queue.append(curroot.rchild)
    def preorder(self,node):
        if node is None:
            return
        print(node.elem,end=" ")
        self.preorder(node.lchild)
        self.preorder(node.rchild)
    def inorder(self,node):
        if node is None:
            return
        self.preorder(node.lchild)
        print(node.elem,end=" ")
        self.preorder(node.rchild)
    def postorder(self,node):
        if node is None:
            return
        self.preorder(node.lchild)
        self.preorder(node.rchild)
        print(node.elem,end=" ")

if __name__=="__main__":
     tr=tree()
     tr.add(3)
     print (tr.preorder(tr.root))
     tr.add(4)
     tr.add(6)
     tr.add(9)
     print (tr.breadth_travel())