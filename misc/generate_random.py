import random 

def position(i, n):
    k = int(n**(0.5))
    a = i//k
    b = i%k

    center = [a+0.9*random.random(), 0.2,b+0.9*random.random()]
    tmp = random.random()

    if(((center[0]-4)**2+(center[1]-0.2)**2+center[2]**2)**(0.5) > 0.9):
        if(tmp < 0.8):
            albedo = [random.random()*random.random(),random.random()*random.random(),random.random()*random.random()]
            return (True, "({},{},{}), 0.2, opaque(({},{},{}))".format(*center,*albedo))
        elif(tmp < 0.95):
            albedo = [random.uniform(0.5,1),random.uniform(0.5,1),random.uniform(0.5,1)]
            fuzz = random.uniform(0,0.5)
            return (True, "({},{},{}), 0.2, metal(({},{},{}), {})".format(*center,*albedo,fuzz))
        else:
            return (True, "({},{},{}), 0.2, glass((1,1,1),1.5)".format(*center))

    return (False, None)


n = int(input())

print("sphere = (0,-1000,0), 1000, opaque((0.5,0.5,0.5))\n")

l = 1

if(n >= 4):
    l = 4
    print("sphere = (0,1,0), 1.0, glass((1,1,1), 1.5)\n")
    print("sphere = (-4,1,0), 1.0, opaque((0.4,0.2,0.1))\n")
    print("sphere = (4,1,0), 1.0, metal((0.7,0.6,0.5), 0.0)\n")

counter = 0
while(counter < n-l):
    is_ok, text_out = position(counter, n)

    if(is_ok):
        counter += 1
        print("sphere = {}\n".format(text_out))
