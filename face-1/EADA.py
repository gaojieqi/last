import numpy as np
import math
import matplotlib.pyplot as plt
import random
import copy


Ns=2#level count
Nh=2#hot count
Nc=2#cold count
delta_T=10#pinch temperature
population=5
a_cost=49000#fix capital
b_cost=2520#
c_cost=0.8#power efficient
i_cost=0.1#interest rate
t_cost=20#operational interval
c_hu=28000#cost of hot_utility
c_cu=30000#cost of cold_utility
decay_rate=0.6#parameter of repair operator
heat_coe=750#heat coefficiency
CF=0.8#coefficient in EADA

hot=[[100,70,300,0,[0.3,0.4]],[140,50,500,0,[0.25,0.44]]]
cold=[[15,30,500,0,[0.55,0.66]],[10,35,300,0,[0.77,0.88]]]

for flow in hot:
    a=float(flow[0])
    b=float(flow[1])
    c=float(flow[2])
    flow[3]=round(c/(a-b),2)
for flow in cold:
    a = float(flow[1])
    b = float(flow[0])
    c = float(flow[2])
    flow[3] = round(c/(a-b), 2)
cold_duty_max = []
hot_duty_max = []
for jj in range(Nh):
    cold_duty_max.append(0)
#Outter iteration
def GA(mut=0.8,crossp=0.7,popsize=20,its_GA=5,its_EADA=10):
    global_fitness=10000000000000000
    pop=[]#no level discrimination
    pop_level=[]#level discrimination
    for ii in range(population):
        pop_unit_level=[]
        pop_unit = []
        for kk in range(Ns):
            pop_level_cache=[]
            for ii in range(Nc):
                for jj in range(Nh):
                    if hot[jj][0]-delta_T>cold[ii][0]:
                        pop_level_cache.append(random.randrange(0,2,1))
                        pop_unit.append(random.randrange(0,2,1))
                    else:
                        pop_level_cache.append(0)
                        pop_unit.append(0)
            pop_unit_level.append(pop_level_cache)
        pop_level.append(pop_unit_level)#for the first unit [[1,0],[1,1]]:[1,0]->Unit 1 level 1,[1,1]->Unit 1 level 2
        pop.append(pop_unit)
    #EADA first iteration
    fitness=[]
    eada_struct = []
    for ii in range(popsize):
        structure_eada, fttt = EADA(pop[ii], mut=0.8, crossp=0.7, popsize=3, its=its_EADA)
        fitness.append(fttt)
        eada_struct.append(structure_eada) #return fitness, len(fitness)==popsize

    #GA iteration
    for kkk in range(its_GA):
        # generate probabilities
        probability = []
        for ii in range(popsize):
            probability.append(float(fitness[ii]) / sum(fitness))
        pro_range=[0]
        for ii in range(popsize):
            sumation=0
            for iii in range(0,ii+1):
                sumation+=probability[iii]
            pro_range.append(sumation)
        # generate parents
        parent = []
        for jj in range(popsize):
            point_pa=random.random()
            for jjj in range(popsize):
                if point_pa>pro_range[jjj] and point_pa<pro_range[jjj+1]:
                    parent.append(pop[jjj])
                    break

        # generate children
        children = []
        for ii in range(popsize):#each iteration only generate one children
            # 1.cross
            parent_a = parent[random.randrange(0, popsize, 1)]
            parent_b = parent[random.randrange(0, popsize, 1)]
            if random.random() < crossp:
                point_a = random.randrange(0, Nh * Nc * Ns - 1, 1)
                point_b = random.randrange(point_a + 1, Nh * Nc * Ns, 1)
                for iii in range(point_b - point_a):
                    parent_a[point_a + iii] = parent_b[point_a + iii]
            # 2.mutation
            for iii in range(Nh * Nc * Ns):
                if random.random() < mut:
                    parent_a[iii] = 1 - parent_a[iii]
            # replace parent with children
            children.append(parent_a)
        #evaluate children
        fitness=[]
        eada_struct=[]
        for ii in range(popsize):
            structure_eada,fttt=EADA(children[ii], mut=0.8, crossp=0.7, popsize=3, its=2)
            fitness.append(fttt)
            eada_struct.append(structure_eada)
        #find global fitness, structure and split
        for iiii in range(fitness):
            if fitness[iiii]<global_fitness:
                global_fitness=fitness[iiii]
                structure=children[ii]
                global_eada_struct=eada_struct[iiii]

    return global_fitness,structure,global_eada_struct


#inner iteration
def EADA(structure_info, mut=0.8, crossp=0.7, popsize=10, its=5):
    pr=[]
    pop = []
    fitness=[]
    best_fitness=0
    abandon_record=0
    #initialize
    ##initialize temperature. 0-3 dimension:cold-in/hot-out/cold-out/hot-in (T[ [[0],[1],[2],[3]],[...] ])
    for ppppp in range(popsize):
        T = []
        for ii in range(Ns):
            T_unit = []
            T_i = []
            for jj in range(Nc):
                T_i.append(0)
            T_unit.append(T_i)
            T_i = []
            for jj in range(Nh):
                T_i.append(0)
            T_unit.append(T_i)
            T_i = []
            for jj in range(Nc):
                T_i.append(0)
            T_unit.append(T_i)
            T_i = []
            for jj in range(Nh):
                T_i.append(0)
            T_unit.append(T_i)
            T_i = []
            T.append(T_unit)

        ##initialize temperature and heat_load
        heat_load = []
        cold_utility = []
        hot_utility = []
        split = []
        heat_load_split = []

        for kk in range(Ns):
            hl = []
            for ii in range(Nc):
                hl.append(0)
            heat_load.append(hl)
        for kk in range(Ns):
            hl = []
            for ii in range(Nc):
                for jj in range(Nh):
                    hl.append(0)
            heat_load_split.append(hl)
        for kk in range(Ns):
            split_unit = []
            tab = []
            t_h_max = []
            for jj in range(Nh):
                tab.append(0)
            for jj in range(Nh):
                tab.append(0)
            # initialize duty and cold stream temperature
            if kk == 0:
                # kk==0
                for ii in range(Nc):
                    T[kk][0][ii] = copy.deepcopy(cold[ii][0])
                for jj in range(Nh):
                    temp_max = 0
                    flag_ini = 0
                    for ii in range(Nc):
                        if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                            temp_temp = cold[ii][0] + delta_T
                            flag_ini = 1
                            if temp_temp > temp_max:
                                temp_max = temp_temp
                    if flag_ini == 1:
                        if temp_max>hot[jj][1]:
                            llll = temp_max - hot[jj][1]
                            cold_utility.append(random.randrange(int(float(llll) * hot[jj][3]), int(hot[jj][2]), 1))
                            T[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                        if temp_max<=hot[jj][1]:
                            cold_utility.append(random.randrange(0, int(hot[jj][2]), 1))
                            T[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                    if flag_ini == 0:
                        tab[jj] = 1  # mark stream with no heat exchange in k level
                        cold_utility.append(random.randrange(0, int(hot[jj][2]), 1))
                        T[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                        T[kk][3][jj] = T[kk][1][jj]
            if kk != Ns - 1:
                # common
                sp = []
                for ii in range(Nc):
                    split_sum = 0
                    split_unit=[]
                    for jj in range(Nh):
                        if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                            ppp = random.random()
                            split_sum += ppp
                            split_unit.append(ppp)
                        else:
                            split_unit.append(0)
                    for jj in range(Nh):
                        if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                            split_unit[jj] = (1 - split_sum+split_unit[jj])
                            break
                    heat_load[kk][ii] = random.random() * cold[ii][2]
                    T[kk][2][ii] = T[kk][0][ii]+ float(heat_load[kk][ii]) / cold[ii][3]
                    sp.append(split_unit)
                split.append(sp)
            if kk == Ns - 1:
                for jj in range(Nh):
                    T[kk][3][jj] = hot[jj][0]
                    T[kk][1][jj] = T[kk - 1][3][jj]
                for ii in range(Nc):
                    T[kk][0][ii]=T[kk-1][2][ii]
                # equality constrain (energy)----split the hot stream duty to cold stream
                sp = []
                for jj in range(Nh):
                    split_sum = 0
                    split_unit=[]
                    for ii in range(Nc):
                        if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                            ppp = random.random()
                            split_sum += ppp
                            split_unit.append(ppp)
                        else:
                            split_unit.append(0)
                    for ii in range(Nc):
                        if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                            split_unit[ii] = (1 - split_sum+split_unit[ii])
                            break
                    heat_load[kk][jj] = float(T[kk][3][jj] - T[kk][1][jj]) * hot[jj][3]
                    T[kk][2][ii] = T[kk][0][ii] + float(heat_load[kk][ii]) / cold[ii][3]
                    sp.append(split_unit)
                split.append(sp)  # --------------split[Ns-1] represent split of hot stream in level k
            # initialize hot stream temperature
            if kk == 0:
                for jj in range(Nh):
                    llkkjj=0
                    for ii in range(Nc):
                        if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                            llkkjj+=(float(heat_load[kk][ii] * split[kk][ii][jj]) / hot[jj][3])
                    T[kk][3][jj] =T[kk][1][jj]+llkkjj
            if kk!=0 and kk!=Ns-1:
                for jj in range(Nh):
                    T[kk][1][jj] = T[kk - 1][3][jj]
                    kklljj=0
                    for ii in range(Nc):
                        if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                            kklljj+=(float(heat_load[kk][ii] * split[kk][ii][jj]) / hot[jj][3])
                    T[kk][3][jj] =T[kk][1][jj]+kklljj
        #repair
        T, heat_load, cold_utility,abandon = repair(T, split, structure_info, heat_load, cold_utility)
        if abandon==1:
            popsize-=1#generate a new one
            abandon_record+=1
            continue
        hot_utility=hot_u_fun(structure_info,heat_load)
        #add to fitness
        fit_unit=fobj(T,split,structure_info,heat_load,cold_utility,hot_utility)

        if len(fitness)==0:
            fitness.append(fit_unit)
        if len(fitness) == 1:
            if fit_unit > fitness[0]:
                fitness=[fit_unit,fitness[0]]
            else:
                fitness.append(fit_unit)
        if len(fitness)>=2:
            if fit_unit > fitness[0]:
                fitness = change(fitness, fit_unit, 0)
                # convert to genetic code
                gen = []
                gen = gen + [split] + [heat_load] + [cold_utility]  # code components
                pop = change(pop, gen, 0)
            else:
                for qwer in range(len(fitness)):
                    if qwer < len(fitness) - 1:
                        if fit_unit < fitness[qwer] and fit_unit > fitness[qwer + 1]:
                            fitness = change(fitness, fit_unit, qwer+1)
                    if qwer == len(fitness)-1:
                        fitness.append(fit_unit)
                        # convert to genetic code
                    gen = []
                    gen = gen + [split] + [heat_load] + [cold_utility]  # code components
                    pop = change(pop, gen, 0)
                    break
    for asdf in range(its):
        ssssss=0
        for ppppp in range(popsize):
            ssssss+=fitness[ppppp]
        probability=[]
        for ppppp in range(popsize):
            probability.append(float(fitness[ppppp])/ssssss)
        pro_range = [0]
        for ii in range(popsize):
            sumation = 0
            for iii in range(0, ii + 1):
                sumation += probability[iii]
            pro_range.append(sumation)
        for ppppp in range(popsize):
            # let pb be the better one, find the right direction
            point_pa=random.randrange(int(popsize/10),int(popsize/90),1)
            pa = copy.deepcopy(pop[point_pa])
            point_pb=random.randrange(0,int(popsize/10),1)
            pb = copy.deepcopy(pop[point_pb])
            buf_pa=getnewList(pa)
            buf_pb=getnewList(pb)
            differ=[]
            for mmmmm in range(len(buf_pa)):
                differ.append(CF*(buf_pb[mmmmm]-buf_pa[mmmmm]))
            differ=mutation(differ,mut)
            #mutation
            buf_pop=getnewList(pop[ppppp])
            diff_buf=[]
            for mmmmm in range(len(differ)):
                diff_buf.append(differ[mmmmm]+buf_pop[mmmmm])
            pop[ppppp]=copy.deepcopy(diff_buf)
            #repair split
            for kk in range(Ns):
                if kk!=Ns-1:
                    for ii in range(Nc):
                        qwer=0
                        for jj in range(Nh):
                            if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                                if pop[ppppp][0][kk][ii][jj]<=0:
                                    pop[ppppp][0][kk][ii][jj]=0
                                qwer+=pop[ppppp][0][kk][ii][jj]
                        for jj in range(Nh):
                            if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1 and qwer!=0:
                                pop[ppppp][0][kk][ii][jj]=pop[ppppp][0][kk][ii][jj]/float(qwer)
                if kk==Ns-1:
                    for jj in range(Nh):
                        qwer=0
                        for ii in range(Nc):
                            if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                                if pop[ppppp][0][kk][jj][ii]<=0:
                                    pop[ppppp][0][kk][jj][ii]=0
                                qwer+=pop[ppppp][0][kk][jj][ii]
                        for ii in range(Nc):
                            if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1 and qwer!=0:
                                pop[ppppp][0][kk][jj][ii]=pop[ppppp][0][kk][jj][ii]/float(qwer)
            #repair heat load
            for kk in range(Ns):
                if kk!=Ns-1:
                    for ii in range(Nc):
                        if pop[ppppp][1][kk][ii]<0:
                            pop[ppppp][1][kk][ii]=0
                if kk == Ns - 1:
                    for jj in range(Nh):
                        if pop[ppppp][1][kk][jj] < 0:
                            pop[ppppp][1][kk][ii] = 0
            #recalclate cold utility
            for jj in range(Nh):
                if pop[ppppp][2][jj]>hot[jj][2]:
                    pop[ppppp][2][jj]=copy.deepcopy(hot[jj][2]*0.8)
                if pop[ppppp][2][jj]<0:
                    pop[ppppp][2][jj]=0
            #recalculate temperature
            T__=recalculate_T(structure_info,pop[ppppp][2],pop[ppppp][1],pop[ppppp][0])
            #repair all
            T__, pop[ppppp][1], pop[ppppp][2],abandon = repair(T__, pop[ppppp][0], structure_info, pop[ppppp][1], pop[ppppp][2])
            if abandon==1:
                abandon_record+=1
                continue#skip it's fitness calculation
            hot_u=hot_u_fun(structure_info,pop[ppppp][1])
            fttt=fobj(T__,pop[ppppp][0],structure_info,pop[ppppp][1],pop[ppppp][2],hot_u)
            if fttt>best_fitness:
                record=[T__,pop[ppppp][0],structure_info,pop[ppppp][1],pop[ppppp][2],hot_u]
                best_fitness=fttt

            else:
                record=1#nonsense
            pr.append((10**10)/fttt)
    print ('abandon record:',abandon_record)
    plt.plot(pr)
    plt.show()
    return record,(10**10)/best_fitness

def hot_u_fun(structure_info,heat_load):
    love = []
    hot_utility=[]
    for ii in range(Nc):
        love.append(0)
        for jj in range(Nh):
            if structure_info[Nh * Nc * (Ns-1) + ii * Nh + jj] == 1:
                love[ii] += heat_load[Ns - 1][jj]

    for ii in range(Nc):
        lkjlkj = 0
        for kk in range(Ns):
            if kk != Ns - 1:
                lkjlkj += heat_load[kk][ii]
            if kk == Ns - 1:
                lkjlkj += copy.deepcopy(love[ii])
        hot_utility.append(cold[ii][2] - lkjlkj)
    return hot_utility
def recalculate_T(structure_info,cold_utility,heat_load,split):
    T=[]
    for kk in range(Ns):
        T_=[]
        if kk==0:
            T__=[]
            for ii in range(Nc):
                T__.append(cold[ii][0])
            T_.append(T__)
            T__ = []
            for jj in range(Nh):
                tttt=hot[jj][1]+cold_utility[jj]/float(hot[jj][3])
                T__.append(tttt)
            T_.append(T__)
            T__ = []
            for ii in range(Nc):
                tttt = cold[ii][0] + heat_load[kk][ii] / float(cold[ii][3])
                T__.append(tttt )
            T_.append(T__)
            T__ = []
            for jj in range(Nh):
                d=0
                f__=0
                for ii in range(Nc):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        d+=heat_load[kk][ii]*split[kk][ii][jj]
                        f__=1
                if f__==1:
                    tttt = T_[1][jj] + d / float(hot[jj][3])
                    T__.append(tttt )
                if f__==0:
                    T__.append(T_[1][jj])
            T_.append(T__)
        if kk!=Ns-1 and kk!=0:
            T__ = []
            for ii in range(Nc):
                T__.append(T[kk-1][2][ii])
            T_.append(T__)
            T__ = []
            for jj in range(Nh):
                T__.append(T[kk - 1][3][jj])
            T_.append(T__)
            T__ = []
            for ii in range(Nc):
                tttt = T[kk-1][2][ii] + heat_load[kk][ii] / float(cold[ii][3])
                T__.append(tttt)
            T_.append(T__)
            T__ = []
            for jj in range(Nh):
                d = 0
                f__ = 0
                for ii in range(Nc):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        d += heat_load[kk][ii] * split[kk][ii][jj]
                        f__ = 1
                if f__ == 1:
                    tttt = T[kk - 1][3][jj]+ d / float(hot[jj][3])
                    T__.append(tttt)
                if f__ == 0:
                    T__.append(T_[1][jj])
            T_.append(T__)
        if kk==Ns-1:
            T__ = []
            for ii in range(Nc):
                T__.append(T[kk - 1][2][ii])
            T_.append(T__)
            T__ = []
            for jj in range(Nh):
                T__.append(T[kk - 1][3][jj])
            T_.append(T__)
            T__ = []
            for ii in range(Nc):
                d=0
                f__=0
                for jj in range(Nh):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        d += heat_load[kk][jj] * split[kk][jj][ii]
                        f__ = 1
                if f__ == 1:
                    tttt = T[kk - 1][2][ii]+ d / float(cold[ii][3])
                    T__.append(tttt)
                if f__ == 0:
                    T__.append(T_[1][ii])
            T_.append(T__)
            T__ = []
            for jj in range(Nh):
                T__.append(hot[jj][0])
            T_.append(T__)
        T.append(T_)
    return T
def repair(t,sp,structure_info,heat_load,cold_utility):
    global_stop=0
    count=0
    abandon=0
    limit=30#maximun count
    while global_stop==0:
        count+=1
        global_flag=1
        for kk in range(Ns):
            if kk==0:
                stop=0
                while stop == 0:
                    count+=1
                    flag = 1
                    for jj in range(Nh):
                        temp_max = 0
                        flag_ini = 0
                        for ii in range(Nc):
                            if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                                temp_temp = cold[ii][0] + delta_T
                                flag_ini = 1
                                if temp_temp > temp_max:
                                    temp_max = temp_temp
                        for ii in range(Nc):
                            if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                                if t[kk][1][jj] - delta_T < t[kk][0][ii] or t[kk][1][jj]>hot[jj][0] or t[kk][1][jj]<hot[jj][1]:
                                    if flag_ini == 1:
                                        llll = temp_max - hot[jj][1]
                                        cold_utility[jj] = random.randrange(int(float(llll) * hot[jj][3]),
                                                                            int(hot[jj][2]), 1)
                                        t[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                                    if flag_ini == 0:
                                        cold_utility[jj] = random.randrange(0, int(hot[jj][2]), 1)
                                        t[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                                        t[kk][3][jj] = t[kk][1][jj]
                                    flag = 0
                                    global_flag=0
                        if t[kk][1][jj]>hot[jj][0] or t[kk][1][jj]<hot[jj][1]:
                            if flag_ini == 1:
                                llll = temp_max - hot[jj][1]
                                cold_utility[jj] = random.randrange(int(float(llll) * hot[jj][3]),
                                                                    int(hot[jj][2]), 1)
                                t[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                            if flag_ini == 0:
                                cold_utility[jj] = random.randrange(0, int(hot[jj][2]), 1)
                                t[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                                t[kk][3][jj] = t[kk][1][jj]
                    if count > limit:
                        stop=1
                    if  flag == 1:
                        stop = 1
            if kk != Ns - 1:
                stop = 0
                for ii in range(Nc):
                    flag_ini = 0
                    for jj in range(Nh):
                        if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                            flag_ini = 1
                    if flag_ini == 0:
                        t[kk][2][ii] = t[kk][0][ii]
                while stop == 0:
                    count += 1
                    flag = 1
                    for ii in range(Nc):
                        for jj in range(Nh):
                            if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                                if t[kk][3][jj] - delta_T < t[kk][2][ii] or t[kk][3][jj] > hot[jj][0] or t[kk][2][ii] > \
                                        cold[ii][1]:
                                    lkjm=heat_load[kk][ii] * (1-decay_rate)
                                    heat_load[kk][ii] = heat_load[kk][ii] * decay_rate
                                    for jjj in range(Nh):
                                        if structure_info[Nh*Nc*kk+ii * Nh + jjj] == 1:
                                            t[kk][3][jjj]-=lkjm * sp[kk][ii][jjj] / float(hot[jjj][3])
                                    t[kk][2][ii] = t[kk][0][ii] + float(heat_load[kk][ii]) / cold[ii][3]
                                    global_flag=0
                                    flag=0
                    if  flag == 1:
                        for ii in range(Nc):
                            t[kk][2][ii] = t[kk][0][ii] + float(heat_load[kk][ii]) / cold[ii][3]
                        for jj in range(Nh):
                            suff = 0
                            for ii in range(Nc):
                                if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                                    suff += heat_load[kk][ii] * sp[kk][ii][jj] / float(hot[jj][3])
                            t[kk][3][jj] = t[kk][1][jj]+suff
                        stop = 1
                    if count > limit:
                        stop=1
                for jj in range(Nh):
                    t[kk+1][1][jj]=t[kk][3][jj]
                for ii in range(Nc):
                    t[kk+1][0][ii]=t[kk][2][ii]
            if kk == Ns - 1:
                for jj in range(Nh):
                    heat_load[kk][jj] = float(t[kk][3][jj] - t[kk][1][jj]) * hot[jj][3]
                for ii in range(Nc):
                    t[kk][2][ii] = t[kk][0][ii] + float(heat_load[kk][ii]) / cold[ii][3]
                stop = 0
                summmm = []
                for jj in range(Nh):
                    summmm.append(0)
                while stop == 0:
                    count += 1
                    flag = 1
                    for jj in range(Nc):
                        for ii in range(Nh):
                            if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                                if t[kk][3][jj] - delta_T < t[kk][2][ii] or t[kk][2][ii]>cold[ii][1]:
                                    ll = (hot[jj][0] - t[kk][1][jj]) * (hot[jj][3])/2
                                    summmm[jj] += ll
                                    cold_utility[jj] += ll
                                    heat_load[kk][jj] -= ll
                                    for iii in range(Nc):
                                        if structure_info[Nh*Nc*kk+iii * Nh + jj] == 1:
                                            t[kk][2][iii] -=ll * sp[kk][jj][iii] / float(cold[iii][3])
                                    flag = 0
                                    global_flag=0
                    if count > limit:
                        stop=1
                    if flag == 1:
                        stop = 1
                for kkk in range(Ns):
                    for jj in range(Nh):
                        ppppp = summmm[jj] / float(hot[jj][3])
                        if kkk!=Ns-1:
                            t[kkk][1][jj] += ppppp
                            t[kkk][3][jj] += ppppp
                        if kkk==Ns-1:
                            t[kkk][1][jj]+=ppppp
        if count>limit:
            global_stop=1
            abandon=1
        if global_flag==1:
            global_stop=1
    return t,heat_load,cold_utility,abandon
def fobj(T,split,structure_info,heat_load,cold_utility,hot_utility):
    c_capital=0
    CU=0
    HU=0
    #equipment cost
    for kk in range(Ns):
        if kk!=Ns-1:
            for ii in range(Nc):
                for jj in range(Nh):
                    if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                        delta_hot=T[kk][3][jj]-T[kk][2][ii]
                        delta_cold=T[kk][1][jj]-T[kk][0][ii]
                        A=heat_load[kk][ii]*split[kk][ii][jj]/float(heat_coe*delta_T_fun(delta_hot,delta_cold))
                        c_capital+=(a_cost+b_cost*(A))*1.5/20
        if kk==Ns-1:
            for jj in range(Nh):
                for ii in range(Nc):
                    if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                        delta_hot = T[kk][3][jj] - T[kk][2][ii]
                        delta_cold = T[kk][1][jj] - T[kk][0][ii]
                        A = heat_load[kk][jj] * split[kk][jj][ii] / float(heat_coe * delta_T_fun(delta_hot, delta_cold))
                        c_capital += (a_cost + b_cost * (A ))*1.5 /20
    #utility cost
    #TODO set different cost of utility according to their qualities
    for jj in range(Nh):
        CU+=cold_utility[jj]
    for ii in range(Nc):
        HU+=hot_utility[ii]
    c_energy=HU*c_hu+CU*c_cu
    c_global=c_energy+c_capital
    return 10**10/float(c_global)
def delta_T_fun(delta_hot,delta_cold):
    if delta_hot>1.7*delta_cold:
        result = (delta_hot - delta_cold) / (math.log(delta_hot) - math.log(delta_cold))
    else:
        result= (delta_hot + delta_cold) / 2
    result=abs(result)
    return round(result,2)
def getnewList(newlist):
    d = []
    for element in newlist:
        if not isinstance(element, list):
            d.append(np.array(element))
        else:
            d.append(np.asarray(getnewList(element), object))
    return d
def mutation(newarray,mut):
    d=[]
    for element in newarray:
        if not isinstance(element, np.ndarray):
            point_pa = random.random()
            if point_pa < mut:
                d.append(element)
            else:
                d.append(0)
        else:
            d.append(np.asarray(mutation(element,mut),object))
    return d
def change(fitness,fit_unit,qwer):
    d=[]
    for i in range(qwer):
        d.append(fitness[i])
    d.append(fit_unit)
    for i in range(qwer,len(fitness)):
        d.append(fitness[i])
    return d


re,fit=EADA(structure_info=[1,1,1,0,1,0,1,1])
print (fit)
