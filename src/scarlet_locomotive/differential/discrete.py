def diff(t, x):
    #if both lists have the same number of items
    if len(t) != len(x):
        print("lists must be the same length")
        return []

    #make an empty list 
    v = []

    #Loop
    for i in range(1, len(t)):
        #calculate difference 
        chg_t = t[i] - t[i - 1]
        chg_x = x[i] - x[i - 1]

        #calculate derivative
        speed = chg_x / chg_t
        v.append(speed)

    return v
