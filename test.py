class A():
    def __init__(self):
        self.aa =1
        self.bb =12

    def __str__(self):
        return str(self.__dict__)
if __name__ == '__main__':
    a = ['1','2','33']
    b = ['2']
    aa = []
    for i in a:
        if i not in b:
            aa.append(i)
    print(aa)