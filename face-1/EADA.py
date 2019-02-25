import numpy as np
import math
import matplotlib.pyplot as plt
import random
import copy


Ns=2#level count
delta_T=1#pinch temperature
a_cost=49000#fix capital(CNY)
b_cost=2520#(CNY)
c_cost=0.8#power efficient
i_cost=0.04#interest rate
t_cost=20#operational interval(year)
c_hu=28#cost of hot_utility(CNY/GJ)
c_cu=10#cost of cold_utility(CNY/GJ)
decay_rate=0.99#parameter of repair operator
heat_coe=1#heat coefficiency-----KW/(K*m^2)
day_adt=2000#ADt per day

CF=0.05#coefficient in EADA
CF_=0.1#coefficienct in water network
upper_range=0.1#Top generation chozen in eade
lower_range=0.7#Worest generaton chozen in eade
selection_rate=0.99#selection of each iteration in eade
factor=day_adt*(10**6)/(3600*24)#scale to energy consumption in second(KJ/S)

water_capacity=0.0042#specia heat capacity for water(GJ/ton)
p_fresh=3.4#price of fresh water(RMB/ton)
PU_count=5#process count
waste_water_count=1#water network output count
fresh_t=15#fresh water temperature
waste_t=30#waste water temperature



hot_origin=[[1600,70,0.26,0],[120,90,0.12,0],[900,160,0.18,0],[850,120,2.9,0],[454,148,0.75,0],[600,250,0.6,0],[600,250,0.25,0],[105,93,0.25,0],[128,120,0.78,0],[109,107,0.65,0],[64,30,2.3,0]]#[inlet_temperature, outlet_temperature,enthalpy(GJ),k]
cold_origin=[[250,600,0.6,0],[148,600,1,0],[90,540,0.14,0],[25,110,0.09,0],[120,151,2.8,0],[10,120,1,0],[85,165,0.6,0],[120,201,0.3,0],[95,124,0.1,0],[21.5,70,0.2,0],[50,60,0.17,0],[15,50,0.96,0],[15,62,0.132,0],[15,60,0.13,0]]
PU=[[62,62,0.7],[60,60,4.2],[70,70,10],[50,50,6.5],[60,60,0.7]]#[inlet_temperature, outlet_temperature, mass_quantity(tone)]
index=[[0,0,1,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1],[0,1,0,0,0,1],[1,1,1,1,1,1]]#index::1-wood preparation;2-washing;3-bleaching;4-pulp machine;5-black liquor evaporation;6-sewer

def water_network(popsize=5,its=10):
    #initialize
    pr=[]
    global_fit=0
    global_structure = []
    global_water_split = []
    #mass split
    PU_split=[]
    for pp in range(PU_count):
        tt=PU[pp][2]
        ttt=split(tt,pp)
        PU_split.append(ttt)
    PU_split,fresh_split=check_water_split(PU_split)
    #initalize EADE pop
    pop=[]
    for qqqqq in range(popsize):
        # calculate stream information and add to hot/cold
        hot,cold=gen_hot_cold(PU_split,fresh_split)
        fit, structure, global_eada_struct = GA(hot, cold)
        #add cost of fresh water
        co=(10 ** 10) /fit
        co+=sum(fresh_split)*p_fresh
        fit=(10 ** 10) /co
        pop.append([PU_split,fit])
    pop.sort(key=lambda x: x[1], reverse=True)
    for iterate in range(its):
        # if int(popsize / 10) == 0:
        #     print('Water network: Not enough popsize')
        #     return 0
        for pppppp in range(popsize):
            point_a = random.randrange(0, int(popsize / 10), 1)
            point_b = random.randrange(popsize - int(popsize / 10), popsize, 1)
            par_a = pop[point_a]
            par_b = pop[point_b]
            diff = []
            buf_a = getnewList(par_a)
            buf_b = getnewList(par_b)
            for mm in range(len(buf_a)):
                diff.append(CF_ * (buf_a[mm] - buf_b[mm]))
            diff = mutation(diff,0.8)
            bbb = pop[pppppp]
            dif_bu=[]
            for mm in range(len(diff)):
                dif_bu.append(bbb[mm]+diff[mm])
            dif_bu[0],fresh_split_=check_water_split(dif_bu[0])
            hot, cold = gen_hot_cold(dif_bu[0], fresh_split_)
            fit, structure, global_eada_struct = GA(hot, cold)
            # add cost of fresh water
            co = (10 ** 10) / fit
            co += sum(fresh_split_) * p_fresh
            fit = (10 ** 10) / co
            pop[pppppp][0]=dif_bu[0]
            pop[pppppp][1] = fit
            pop.sort(key=lambda x: x[1], reverse=True)
            if fit>global_fit:
                global_fit=fit
                global_structure=[structure,global_eada_struct]
                global_water_split=[dif_bu[0],fresh_split_]
            pr.append((10 ** 10) / global_fit)
        #eliminate unsuitable units
        # pop=select(pop)
        # popsize=len(pop)

        print('--Water allocation iterate:', iterate)
    plt.plot(pr)
    plt.show()
    return 10**10/global_fit,global_structure,global_water_split
def GA(hot,cold,mut=0.95,crossp=0.6,popsize=5,its_GA=20):
    #add slop
    for flow in range(len(hot)):
        a = float(hot[flow][0])
        b = float(hot[flow][1])
        c = float(hot[flow][2])
        hot[flow][3] = c / (a - b)
    for flow in range(len(cold)):
        a = float(cold[flow][0])
        b = float(cold[flow][1])
        c = float(cold[flow][2])
        cold[flow][3] = c / (b - a)
    #initialize
    Nh=len(hot)
    Nc=len(cold)
    pr=[]
    global_fitness=0
    pop=[]#no level discrimination
    structure=0
    global_eada_struct=0
    for ppp in range(popsize):
        pop_unit = []
        for kk in range(Ns):
            for ii in range(Nc):
                for jj in range(Nh):
                    if hot[jj][0]-delta_T>cold[ii][0]:
                        pop_unit.append(random.randrange(0,2,1))
                    else:
                        pop_unit.append(0)
        pop.append(pop_unit)
    #EADA first iteration
    fitness=[]
    eada_struct = []
    for ii in range(popsize):
        structure_eada, fttt = EADA(hot,cold,pop[ii])
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
        # generate children
        tag = []
        # generate parents
        parent = []
        for jj in range(2):
            point_pa = random.random()
            for jjj in range(popsize):
                if point_pa > pro_range[jjj] and point_pa < pro_range[jjj + 1]:
                    parent.append(pop[jjj])
                    tag.append(jjj)
                    break
        # 1.cross
        parent_a = parent[0]
        parent_b = parent[1]
        if random.random() < crossp:
            point_a = random.randrange(0, Nh * Nc * Ns - 1, 1)
            point_b = random.randrange(point_a + 1, Nh * Nc * Ns, 1)
            for iii in range(point_b - point_a):
                kkkkk = parent_b[point_a + iii]
                parent_b[point_a + iii] = parent_a[point_a + iii]
                parent_a[point_a + iii] = kkkkk
        # 2.mutation
        for iii in range(Nh * Nc * Ns):
            if random.random() > mut:
                parent_a[iii] = 1 - parent_a[iii]
            if random.random() > mut:
                parent_b[iii] = 1 - parent_b[iii]
        # 3.repaire constrains
        for iii in range(Nh*Nc*Ns):
            if parent_a[iii]==1:
                iiii=int((iii%Ns)/Nc)
                jjjj=((iii%Ns)%Nc)-1
                if hot[jjjj][0]-delta_T<cold[iiii][0]:
                    parent_a[iii]=0
            if parent_b[iii]==1:
                iiii = int((iii % Ns) / Nc)
                jjjj = ((iii % Ns) % Nc) - 1
                if hot[jjjj][0]-delta_T<cold[iiii][0]:
                    parent_b[iii] = 0
        #change parents
        pop[tag[0]] = parent_a
        pop[tag[1]] = parent_b
        str_a,f_a=EADA(hot,cold,parent_a)
        str_b,f_b=EADA(hot,cold,parent_b)
        fitness[tag[0]]=f_a
        fitness[tag[1]]=f_b
        eada_struct[tag[0]]=str_a
        eada_struct[tag[1]]=str_b
        #find global fitness, structure and split
        for iiii in range(len(fitness)):
            if fitness[iiii]>global_fitness:
                global_fitness=fitness[iiii]
                structure=pop[iiii]
                global_eada_struct=eada_struct[iiii]
        pr.append((10 ** 10) / global_fitness)
        print("GA iteration=",kkk)
    plt.plot(pr)
    plt.show()
    return (10 ** 10)/global_fitness,structure,global_eada_struct
def EADA(hot,cold,structure_info, mut=0.95, crossp=0.7, popsize=2000, its=5):
    Nh=len(hot)
    Nc=len(cold)
    pop = []
    record=0
    best_fitness=1
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
            if kk!=Ns-1:
                hl = []
                for ii in range(Nc):
                    hl.append(0)
                heat_load.append(hl)
            if kk==Ns-1:
                hl = []
                for ii in range(Nh):
                    hl.append(0)
                heat_load.append(hl)
        for kk in range(Ns):
            hl = []
            for ii in range(Nc):
                for jj in range(Nh):
                    hl.append(0)
            heat_load_split.append(hl)
        for kk in range(Ns):
            # initialize duty and cold stream temperature
            if kk == 0:
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
                            cold_utility.append(ran(float(llll) * hot[jj][3],hot[jj][2]))
                            T[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                        if temp_max<=hot[jj][1]:
                            cold_utility.append(ran(0,hot[jj][2]))
                            T[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                    if flag_ini == 0:
                        cold_utility.append(ran(0,hot[jj][2]))
                        T[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                        T[kk][3][jj] = T[kk][1][jj]
            if kk != Ns - 1:
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
                            split_unit[jj] = split_unit[jj] / split_sum
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
                            split_unit[ii] = split_unit[ii]/split_sum
                    sp.append(split_unit)
                for ii in range(Nc):
                    nnnnn = 0
                    for jj in range(Nh):
                        if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                            nnnnn += heat_load[kk][jj] * sp[jj][ii]
                split.append(sp)  # --------------split[Ns-1] represent split of hot stream in level k
            # initialize hot stream temperature
            if kk == 0:
                for jj in range(Nh):
                    llkkjj=0
                    for ii in range(Nc):
                        if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                            ddddd=heat_load[kk][ii] * split[kk][ii][jj]
                            llkkjj=llkkjj+(float(ddddd) / hot[jj][3])
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
        T, heat_load, cold_utility = repair(hot,cold,T, split, structure_info, heat_load, cold_utility)
        hot_utility=hot_u_fun(hot,cold,structure_info,heat_load)
        #add to fitness
        fit_unit=fobj(hot,cold,T,split,structure_info,heat_load,cold_utility,hot_utility)
        #insert to pop and sort
        # convert to genetic code
        gen = []
        gen = gen + [split] + [heat_load] + [cold_utility]+[fit_unit]  # code components
        pop .append(gen)
        pop.sort(key=lambda x:x[3],reverse=True)
    for asdf in range(its):
        for ppppp in range(popsize):
            #check whether popsize is enough
            if int(popsize * upper_range) == 0 or int(popsize * upper_range) == int(popsize * lower_range):
                # print('EADE not enough popsize')
                break
            # let pb be the better one, find the right direction
            point_pa=random.randrange(int(popsize*upper_range),int(popsize*lower_range),1)
            pa = copy.deepcopy(pop[point_pa])
            point_pb=random.randrange(0,int(popsize*upper_range),1)
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
            #ignore the effect on fitness
            diff_buf[3]=0
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
                            pop[ppppp][1][kk][jj] = 0
            #recalclate cold utility
            for jj in range(Nh):
                if pop[ppppp][2][jj]>hot[jj][2]:
                    pop[ppppp][2][jj]=copy.deepcopy(hot[jj][2]*0.8)
                if pop[ppppp][2][jj]<0:
                    pop[ppppp][2][jj]=0
            #recalculate temperature
            T__=recalculate_T(hot,cold,structure_info,pop[ppppp][2],pop[ppppp][1],pop[ppppp][0])
            #repair all
            T__, pop[ppppp][1], pop[ppppp][2]= repair(hot,cold,T__, pop[ppppp][0], structure_info, pop[ppppp][1], pop[ppppp][2])
            hot_u=hot_u_fun(hot,cold,structure_info,pop[ppppp][1])
            fit_unit=fobj(hot,cold,T__,pop[ppppp][0],structure_info,pop[ppppp][1],pop[ppppp][2],hot_u)
            pop[ppppp][3]=fit_unit
            # insert to pop and sort
            pop.sort(key=lambda x: x[3], reverse=True)
            #find the best one
            if fit_unit>best_fitness:
                record=[T__,pop[ppppp][0],structure_info,pop[ppppp][1],pop[ppppp][2],hot_u]
                best_fitness=fit_unit
            else:
                record=1#nonsense
        #eliminate unsuitable units
        pop=select(pop)
        popsize=len(pop)
    return record,best_fitness
def hot_u_fun(hot,cold,structure_info,heat_load):
    Nh=len(hot)
    Nc=len(cold)
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
def recalculate_T(hot,cold,structure_info,cold_utility,heat_load,split):
    Nh=len(hot)
    Nc=len(cold)
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
                for jj in range(Nh):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        d += heat_load[kk][jj] * split[kk][jj][ii]
                tttt = T[kk - 1][2][ii]+ d / float(cold[ii][3])
                T__.append(tttt)
            T_.append(T__)
            T__ = []
            for jj in range(Nh):
                T__.append(hot[jj][0])
            T_.append(T__)
        T.append(T_)
    return T
def repair(hot,cold,t,sp,structure_info,heat_load,cold_utility):
    Nh=len(hot)
    Nc=len(cold)
    for kk in range(Ns):
        if kk == 0:
            for jj in range(Nh):
                temp_max = 0
                flag_ini = 0
                for ii in range(Nc):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        temp_temp = cold[ii][0] + delta_T
                        flag_ini = 1
                        if temp_temp > temp_max:
                            temp_max = temp_temp
                if t[kk][1][jj] > hot[jj][0] or t[kk][1][jj] < hot[jj][1] or t[kk][1][jj] < temp_max:
                    if flag_ini == 1:
                        llll = temp_max - hot[jj][1]
                        cold_utility[jj] = ran(float(llll) * hot[jj][3], hot[jj][2])
                        t[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                    if flag_ini == 0:
                        cold_utility[jj] = ran(0, hot[jj][2])
                        t[kk][1][jj] = hot[jj][1] + float(cold_utility[jj]) / hot[jj][3]
                        t[kk][3][jj] = t[kk][1][jj]
                    global_flag = 0
        if kk != Ns - 1:
            for ii in range(Nc):
                flag_ini = 0
                for jj in range(Nh):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        flag_ini = 1
                if flag_ini == 0:
                    t[kk][2][ii] = t[kk][0][ii]
            # (1) check t[kk][2](cold side) temperature
            for ii in range(Nc):
                lkjm = 0
                while t[kk][2][ii] > cold[ii][1]:
                    lkjm += heat_load[kk][ii] * (1 - decay_rate)
                    heat_load[kk][ii] = heat_load[kk][ii] * decay_rate
                    t[kk][2][ii] = t[kk][0][ii] + heat_load[kk][ii] / cold[ii][3]
                    t[kk + 1][0][ii] = t[kk][2][ii]
                for jj in range(Nh):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        t[kk][3][jj] -= lkjm * sp[kk][ii][jj] / float(hot[jj][3])
                        t[kk + 1][1][jj] = t[kk][3][jj]
            # (2) check t[kk][3](hot side) temperature
            for jj in range(Nh):
                while t[kk][3][jj] > hot[jj][0]:
                    ccccc = 0
                    for ii in range(Nc):
                        if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                            # 1.decay
                            lkjm = heat_load[kk][ii] * sp[kk][ii][jj] * decay_rate
                            ccccc += heat_load[kk][ii] * sp[kk][ii][jj] * (1 - decay_rate)
                            # 2.recalculate split
                            su = 0
                            for jjj in range(Nh):
                                if jjj != jj:
                                    su += heat_load[kk][ii] * sp[kk][ii][jjj]
                            # 3.spot sum
                            su_buf = su + lkjm
                            for jjj in range(Nh):
                                if jjj != jj:
                                    sp[kk][ii][jjj] = heat_load[kk][ii] * sp[kk][ii][jjj] / su_buf
                                else:
                                    sp[kk][ii][jjj] = lkjm / su_buf
                            heat_load[kk][ii] = su_buf
                            # 4.adjust temperature
                            t[kk][2][ii] = t[kk][0][ii] + heat_load[kk][ii] / cold[ii][3]
                            t[kk + 1][0][ii] = t[kk][2][ii]
                    t[kk][3][jj] -= ccccc / hot[jj][3]
                    t[kk + 1][1][jj] = t[kk][3][jj]
            # (3) check t[kk][3]-t[kk][2] and t[kk+1][1]-t[kk+1][0](delta temperature)
            for ii in range(Nc):
                for jj in range(Nh):
                    # check t[kk][3]-t[kk][2]
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        while t[kk][3][jj] - delta_T < t[kk][2][ii]:
                            # 1.smaller heat_load of cold stream
                            lkjm = heat_load[kk][ii] * (1 - decay_rate)
                            heat_load[kk][ii] = heat_load[kk][ii] * decay_rate
                            for jjj in range(Nh):
                                if structure_info[Nh * Nc * kk + ii * Nh + jjj] == 1:
                                    cold_utility[jjj] += lkjm * sp[kk][ii][jjj]
                            # 2.larger cold_utility use
                            ddddd = hot[jj][0] - t[kk][3][jj]
                            dd = ran(0, ddddd)
                            # 3.adjust hot stream temperature
                            for kkkk in range(Ns):
                                if kkkk != Ns - 1:
                                    t[kkkk][1][jj] += dd
                                    t[kkkk][3][jj] += dd
                                if kkkk == Ns - 1:
                                    t[kkkk][1][jj] += dd
                            # 4.adjust cold_utility
                            cold_utility[jj] = cold_utility[jj] + dd / hot[jj][3]
                            # 5.adjust cold_stream temperature
                            t[kk][2][ii] = t[kk][0][ii] + heat_load[kk][ii] / cold[ii][3]
                            t[kk + 1][0][ii] = t[kk][2][ii]
                    # check t[kk+1][1]-t[kk+1][0]
                    if structure_info[Nh * Nc * (kk + 1) + ii * Nh + jj] == 1:
                        while t[kk + 1][1][jj] - delta_T < t[kk + 1][0][ii]:
                            # 1.smaller heat_load of cold stream
                            lkjm = heat_load[kk][ii] * (1 - decay_rate)
                            heat_load[kk][ii] = heat_load[kk][ii] * decay_rate
                            for jjj in range(Nh):
                                if structure_info[Nh * Nc * kk + ii * Nh + jjj] == 1:
                                    cold_utility[jjj] += lkjm * sp[kk][ii][
                                        jjj]  # keep hot stream temperature[3] the same
                            # 2.larger cold_utility use
                            ddddd = hot[jj][0] - t[kk][3][jj]
                            dd = ran(0, ddddd)
                            # 3.adjust hot stream temperature
                            for kkkk in range(Ns):
                                if kkkk != Ns - 1:
                                    t[kkkk][1][jj] += dd
                                    t[kkkk][3][jj] += dd
                                if kkkk == Ns - 1:
                                    t[kkkk][1][jj] += dd
                            # 4.adjust cold_utility
                            cold_utility[jj] = cold_utility[jj] + dd / hot[jj][3]
                            # 5.adjust cold_stream temperature
                            t[kk][2][ii] = t[kk][0][ii] + heat_load[kk][ii] / cold[ii][3]
                            t[kk + 1][0][ii] = t[kk][2][ii]
        if kk == Ns - 1:
            for jj in range(Nh):
                heat_load[kk][jj] = float(t[kk][3][jj] - t[kk][1][jj]) * hot[jj][3]
            for ii in range(Nc):
                nnnnnn = 0
                for jj in range(Nh):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        nnnnnn += heat_load[kk][jj] * sp[kk][jj][ii]
                t[kk][2][ii] = t[kk][0][ii] + float(nnnnnn) / cold[ii][3]
            # (1) check t[kk][2] (cold stream) temperature
            for ii in range(Nc):
                while t[kk][2][ii] > cold[ii][1]:
                    for jj in range(Nh):
                        if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                            # 1.decay
                            lkjm = heat_load[kk][jj] * sp[kk][jj][ii] * decay_rate
                            ccccc = heat_load[kk][jj] * sp[kk][jj][ii] * (1 - decay_rate)
                            # 2.recalculate split
                            su = 0
                            for iii in range(Nc):
                                if iii != ii:
                                    su += heat_load[kk][jj] * sp[kk][jj][iii]
                            # 3.spot sum
                            su_buf = su + lkjm
                            for iii in range(Nc):
                                if iii != ii:
                                    sp[kk][jj][iii] = heat_load[kk][jj] * sp[kk][jj][iii] / su_buf
                                else:
                                    sp[kk][jj][iii] = lkjm / su_buf
                            heat_load[kk][jj] = su_buf
                            # 4.adjust temperature
                            dd = ccccc / hot[jj][3]
                            for kkkk in range(Ns):
                                if kkkk != Ns - 1:
                                    t[kkkk][1][jj] += dd
                                    t[kkkk][3][jj] += dd
                                if kkkk == Ns - 1:
                                    t[kkkk][1][jj] += dd
                            # 5.adjust cold_utility
                            cold_utility[jj] += ccccc
                            t[kk][2][ii] = t[kk][0][ii] + heat_load[kk][ii] / cold[ii][3]
            # (2) check t[kk][3]-t[kk][2]
            for jj in range(Nh):
                for ii in range(Nc):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        while t[kk][3][jj] - t[kk][2][ii] < delta_T:
                            # 1.decay
                            lkjm = heat_load[kk][jj] * sp[kk][jj][ii] * decay_rate
                            ccccc = heat_load[kk][jj] * sp[kk][jj][ii] * (1 - decay_rate)
                            # 2.recalculate split
                            su = 0
                            for iii in range(Nc):
                                if iii != ii:
                                    su += heat_load[kk][jj] * sp[kk][jj][iii]
                            # 3.spot sum
                            su_buf = su + lkjm
                            for iii in range(Nc):
                                if iii != ii:
                                    sp[kk][jj][iii] = heat_load[kk][jj] * sp[kk][jj][iii] / su_buf
                                else:
                                    sp[kk][jj][iii] = lkjm / su_buf
                            heat_load[kk][jj] = su_buf
                            # 4.adjust temperature
                            dd = ccccc / hot[jj][3]
                            for kkkk in range(Ns):
                                if kkkk != Ns - 1:
                                    t[kkkk][1][jj] += dd
                                    t[kkkk][3][jj] += dd
                                if kkkk == Ns - 1:
                                    t[kkkk][1][jj] += dd
                            # 5.adjust cold_utility
                            cold_utility[jj] += ccccc
                            t[kk][2][ii] = t[kk][0][ii] + heat_load[kk][ii] / cold[ii][3]
    for jj in range(Nh):
        sum_=0
        for kk in range(Ns):
            if kk!=Ns-1:
                for ii in range(Nc):
                    if structure_info[Nh * Nc * kk + ii * Nh + jj] == 1:
                        sum_+=heat_load[kk][ii]*sp[kk][ii][jj]
            if kk==Ns-1:
                sum_+=heat_load[kk][jj]
        cold_utility[jj]=hot[jj][2]-sum_
    return t,heat_load,cold_utility
def fobj(hot,cold,T,split,structure_info,heat_load,cold_utility,hot_utility):
    Nh=len(hot)
    Nc=len(cold)
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
                        A=factor*heat_load[kk][ii]*split[kk][ii][jj]/float(heat_coe*delta_T_fun(delta_hot,delta_cold))
                        # n = int(A / 100)
                        print(A)
                        # TODO A**c_cost
                        c_capital+=(a_cost+b_cost*(A))*2.19/(t_cost*day_adt*365)
        if kk==Ns-1:
            for jj in range(Nh):
                for ii in range(Nc):
                    if structure_info[Nh*Nc*kk+ii * Nh + jj] == 1:
                        delta_hot = T[kk][3][jj] - T[kk][2][ii]
                        delta_cold = T[kk][1][jj] - T[kk][0][ii]
                        A = factor*heat_load[kk][jj] * split[kk][jj][ii] / float(heat_coe* delta_T_fun(delta_hot, delta_cold))
                        # n=int(A/100)
                        print(A)
                        #TODO A**c_cost
                        c_capital += (a_cost + b_cost * (A))*2.19 /(t_cost*day_adt*365)
    #utility cost
    #TODO set different cost of utility according to their qualities
    for jj in range(Nh):
        CU+=cold_utility[jj]
    for ii in range(Nc):
        HU+=hot_utility[ii]
    c_energy=HU*c_hu+CU*c_cu
    print(structure_info)
    print("hot utility:",HU)
    print ("cold utility:",CU)
    print("operation cost:",c_energy)
    print("fix cost:",c_capital)
    c_global=c_energy+c_capital
    return 10**10/float(c_global)
def delta_T_fun(delta_hot,delta_cold):
    # if delta_hot>1.7*delta_cold:
    #     result = (delta_hot - delta_cold) / (math.log(delta_hot) - math.log(delta_cold))
    # else:
    result= ((delta_hot*delta_cold)*(delta_hot + delta_cold) / 2)** (1. / 3)
    result=abs(result)
    return result
def ran(start,end):
    if end<=start:
        print ("Random error")
        return 0
    dif=random.random()*(end-start)
    result=start+dif
    return result
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
    for i in range(qwer+1):
        d.append(fitness[i])
    d.append(fit_unit)
    for i in range(qwer,len(fitness)):
        d.append(fitness[i])
    return d
def abandon_fun(pop,ppppp):
    d=[]
    for i in range(len(pop)):
        if i!=pop:
            d.append(pop[i])
    return d
def select(pop):
    li=int(len(pop)*selection_rate)
    d=[]
    for i in range(li):
        d.append(pop[i])
    return d
def split(total,ind):#index::1-wood preparation;2-washing;3-bleaching;4-pulp machine;5-black liquor evaporation
    s_buf = [0]
    num=sum(index[ind])
    for pp in range(num - 1):
        flag = 1
        while flag == 1:
            aaaaaa = random.random()*total
            if aaaaaa not in s_buf:
                s_buf.append(aaaaaa)
                flag = 0
    s_buf.sort()
    s_buf.append(total)
    split_ = []
    ffff=0
    for pp in range(PU_count+waste_water_count):
        if index[ind][pp]==1:
            split_.append(s_buf[ffff + 1] - s_buf[ffff])
            ffff+=1
        else:
            split_.append(0)
    return split_
def mix_T(m,t):
    if len(m)==1:
        return m[0],t[0]
    else:
        aaaaa,bbbbb=mix_T(m[1:],t[1:])
        return sum(m),(m[0]*t[0]+aaaaa*bbbbb)/(m[0]+aaaaa)
def check_water_split(PU_split):
    fresh_split=[]
    for pp in range(len(PU_split)):
        s=0
        for ppp in range(len(PU_split)):
            if PU_split[ppp][pp]<0:
                PU_split[ppp][pp]=0
            s+=PU_split[ppp][pp]
        if s>=PU[pp][2]:
            fresh_split.append(0)
            for ppp in range(len(PU_split)):
                PU_split[ppp][pp] *= PU[pp][2] / s
        if s < PU[pp][2]:
            fresh_split.append(PU[pp][2]-s)
    return PU_split,fresh_split
def gen_hot_cold(PU_split,fresh_split):
    hot = copy.deepcopy(hot_origin)
    cold = copy.deepcopy(cold_origin)
    m_wast = []
    t_wast = []
    for pp in range(len(PU)):
        s = 0
        m = []
        t = []
        for jj in range(len(PU_split)):
            s += PU_split[jj][pp]
        s += fresh_split[pp]
        for jj in range(len(PU_split)):
            PU_split[jj][pp] *= PU[pp][2] / s
            m.append(PU_split[jj][pp])
            t.append(PU[jj][1])
        fresh_split[pp] *= PU[pp][2] / s
        m.append(fresh_split[pp])
        t.append(fresh_t)
        #calculate temperature of flow
        m,t_start = mix_T(m, t)
        t_end = PU[pp][0]
        if t_start > t_end:
            enl = (t_start - t_end) * PU[pp][2] * water_capacity
            stream = [t_start, t_end, enl, 0]
            hot.append(stream)
        if t_start < t_end:
            enl = (-t_start + t_end) * PU[pp][2] * water_capacity
            stream = [t_start, t_end, enl, 0]
            cold.append(stream)
    for pp in range(PU_count):
        s=0
        for ppp in range(PU_count):
           s+=PU_split[pp][ppp]
        sewer_mass=PU[pp][2]-s
        t_wast.append(PU[pp][1])
        m_wast.append(sewer_mass)
    #stream in sewer_waste
    m,t_start=mix_T(m_wast,t_wast)
    t_end=waste_t
    if t_start > t_end:
        enl = (t_start - t_end) * sum(m_wast) * water_capacity
        stream = [t_start, t_end, enl, 0]
        hot.append(stream)
    if t_start < t_end:
        enl = (-t_start + t_end) * sum(m_wast) * water_capacity
        stream = [t_start, t_end, enl, 0]
        cold.append(stream)
    return hot,cold
def recalculate_split(ii,jj,kk,heat_load,sp,Nh,Nc):
    if kk!=Ns-1:
        lkjm = heat_load[kk][ii] * sp[kk][ii][jj] * decay_rate
        su = 0
        for jjj in range(Nh):
            if jjj != jj:
                su += heat_load[kk][ii] * sp[kk][ii][jjj]
        # spot sum
        su_buf = su + lkjm
        for jjj in range(Nh):
            if jjj != jj:
                sp[kk][ii][jjj] = heat_load[kk][ii] * sp[kk][ii][jjj] / su_buf
            else:
                sp[kk][ii][jjj] = lkjm / su_buf
        heat_load[kk][ii] = su_buf

    return heat_load,sp
###main part



global_cost,structure,global_eada_struct=GA(hot_origin,cold_origin)
print (global_cost)
# cost,aaa,bbbb=water_network()
# print ('Cost is:',cost)
# print ('Structure:',aaa)
# print ('Water split:',bbbb)
