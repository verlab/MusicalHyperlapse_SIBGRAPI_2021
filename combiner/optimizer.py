from header import *
import dijkstra
from scipy import interpolate
from skimage import measure
import combiner.thirdpart
from combiner.thirdpart import dtw

def calc_optim_w(song_len,video_len):
    w_max = global_optimizer_w
    w = min(w_max,max(4,int(2*video_len/song_len)))
    return w

def euclidean_distance(p1,p2):
    dist = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return dist

def calc_similarity_sv(song_point,video_point,mode="disc"):

    s_v = song_point[1]*0.5
    s_a = song_point[2]*0.5

    if(mode=="cont"):
        v_v = video_point[1]
        v_a = video_point[2]
    elif(mode=="disc"):
        v_v = video_point[3]
        v_a = video_point[4]

    max_dist = 2.828427 #Distance from [-1,-1] to [+1.+1]
    dist = euclidean_distance([s_v, s_a], [v_v, v_a])
    dist = dist/max_dist
    psim = 1-dist

    return psim

def calc_similarity_vv(img1,img2):
    p_sim = measure.compare_ssim(img1,img2,multichannel=True)
    return p_sim

class optimizer_uniform():

    def optimize(self,song_points,video_points):
        
        n_sngfms = len(song_points)
        n_vidfms = len(video_points)
        n_nxtfms = n_vidfms/n_sngfms

        song_points_f = []
        video_points_f = []

        i_frame_s = 0
        i_frame_v = 0
        while(i_frame_s<n_sngfms and i_frame_v<n_vidfms):
            utils.show_progress("Running uniform optimizer... ",i_frame_s,n_sngfms)
            song_points_f.append(song_points[int(i_frame_s)])
            video_points_f.append(video_points[int(i_frame_v)])
            i_frame_s+=1
            i_frame_v+=n_nxtfms

        return song_points_f, video_points_f

class optimizer_uniform_p():

    def optimize(self,song_points,video_points):

        n_sngfms = len(song_points)
        n_vidfms = len(video_points)
        n_drpfms = n_vidfms-n_sngfms

        n_nxtfms = n_vidfms/n_sngfms

        video_points_f = []
        song_points_f = []

        i_frame_s = 0
        i_frame_v = 0
        while(i_frame_s<n_sngfms):
            utils.show_progress("Running uniform plus optimizer... ",i_frame_s,n_sngfms)
            max_sim = 0
            j_frame_best = i_frame_v
            for j_frame_v in range(int(i_frame_v),min(int(i_frame_v+n_nxtfms),n_vidfms-1)):
                pair_sim = calc_similarity_sv(song_points[i_frame_s],video_points[int(j_frame_v)])
                if(pair_sim>max_sim):
                    max_sim = pair_sim
                    j_frame_best = j_frame_v
            
            video_points_f.append(video_points[int(j_frame_best)])
            song_points_f.append(song_points[i_frame_s])
            
            i_frame_s+=1
            i_frame_v+=n_nxtfms

            if(i_frame_s>n_sngfms):
                break

        return song_points_f, video_points_f

class optimizer_greedy():

    def optimize_acccost(self,song_points,video_points):

        print("Note: Using optimize_acccost (not considering emotion similarity)")

        n_sngfms = len(song_points)
        n_vidfms = len(video_points)
        n_vsdiff = n_vidfms/n_sngfms

        acc_vet = []
        for i in range(len(song_points)):
            song_point = song_points[i]
            acc = song_point[2]
            acc = (acc+1)/2
            acc = int(9*acc)+1
            acc_vet.append(acc)

        video_points_f = []
        song_points_f = song_points

        acc_scl=1

        i_s=0
        i_v=0
        while(i_s<len(song_points) and i_v<len(video_points)):
            utils.show_progress("Running greedy optimizer... ",i_s,len(song_points))
            video_points_f.append(video_points[int(i_v)])
            i_v+=acc_vet[i_s]*acc_scl
            i_s+=1

        return song_points_f, video_points_f 

    def optimize_simcost(self,song_points,video_points):

        n_sngfms = len(song_points)
        n_vidfms = len(video_points)
        n_vsdiff = n_vidfms/n_sngfms

        song_points_f = []
        video_points_f = []

        i_s=0
        i_v=0
        w = calc_optim_w(n_sngfms,n_vidfms)
        while(i_s<len(song_points) and i_v<len(video_points)):
            utils.show_progress("Running greedy optimizer... ",i_s,len(song_points))
            maxsim = calc_similarity_sv(song_points[i_s],video_points[i_v])
            i_maxsim = i_v
            for i_v2 in range(i_v,min(i_v+w,len(video_points))):
                pairsim = calc_similarity_sv(song_points[i_s],video_points[i_v2])
                if(pairsim>maxsim):
                    maxsim = pairsim
                    i_maxsim = i_v2
            song_points_f.append(song_points[i_s])
            video_points_f.append(video_points[i_maxsim])
            i_v = i_maxsim+1
            i_s+=1

        return song_points_f, video_points_f

    def optimize(self,song_points,video_points):
        return self.optimize_simcost(song_points,video_points)

class optimizer_dijkstra():

    def diag_bold_matrix(self,size,radius):
        matrix = np.zeros(size)
        matrix[0][0] = 1
        matrix[matrix.shape[0]-1][matrix.shape[1]-1] = 1
        for i in range(1,matrix.shape[0]-1):
            for j in range(radius):
                a = min(size[1]-1,math.floor(i*size[1]/size[0])+j)
                matrix[i][a] = 1
        return matrix

    def mask_bold(self,matrix,radius):
        boldmatrix = self.diag_bold_matrix(matrix.shape,radius)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = matrix[i][j]*boldmatrix[i][j]
        return matrix

    def get_nodes(self,matrix):
        nodes = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] != 0:
                    nodes.append((i,j))
        return nodes

    def search_next_line_nodes(self,node, matrix):
        next_line_index = min(node[0]+1,matrix.shape[0]-1)
        search_area = []
        for i in range(matrix.shape[1]):
            search_area.append((next_line_index,i))
        return search_area

    def valid_edge(self,edge,nodes,maxjump):
        if edge[1] in nodes and edge[0][1] < edge[1][1] and abs(edge[0][1] - edge[1][1]) <= maxjump:
            return True

    def connect_nodes(self,nodes,matrix,maxjump):
        edges = []
        cont = 0
        for node1 in nodes: 
            utils.show_progress("Connecting nodes... ",cont,len(nodes))
            cont = cont + 1
            next_line_nodes = self.search_next_line_nodes(node1, matrix)
            for node2 in next_line_nodes:
                if self.valid_edge((node1,node2),nodes,maxjump):
                    edges.append((node1,node2))
        return edges

    def separate(self,edges):
        x = []
        y = []
        for i in edges:
            x.append(i[0])
            y.append(i[1])
        return y, x

    def dijkstra_calc(self,edges,S):
        graph = dijkstra.Graph()
        for i in edges:
            cost = 1 - S[i[1][0]][i[1][1]]
            graph.add_edge(i[0], i[1], cost)
        dijkstra_result = dijkstra.DijkstraSPF(graph, (0,0))
        return dijkstra_result.get_path(edges[-1][1])

    def dijkstra_combiner(self,S,radius,maxjump):

        S = self.mask_bold(S,radius)
        nodes = self.get_nodes(S)
        edges = self.connect_nodes(nodes,S,maxjump)
        path = self.dijkstra_calc(edges,S)
        v, m = self.separate(path)

        print(v)

        return m, v

    def compute_similarity(self,song_points,video_points):

        sim_mat = np.zeros((len(song_points),len(video_points)))

        for i in range(len(song_points)):
            utils.show_progress("Computing Similarities... ",i,len(song_points))
            for j in range(len(video_points)):
                sim_mat[i][j] = calc_similarity_sv(song_points[i],video_points[j])

        return sim_mat

    def optimize(self,song_points,video_points):

        print("Running dijkstra optimizer...")

        sim_mat = self.compute_similarity(song_points,video_points)

        w = calc_optim_w(len(song_points),len(video_points))

        song_ids, video_ids = self.dijkstra_combiner(sim_mat,radius=int(w/2),maxjump=100000)

        print("Done")

        song_points_f = []
        video_points_f = []

        for s_id in song_ids:
            song_points_f.append(song_points[s_id])

        for v_id in video_ids:
            video_points_f.append(video_points[v_id])

        return song_points_f, video_points_f

class optimizer_dijkstra_c():

    def optimize(self,song_points,video_points,resize_dim=(128,64)):

        print("Running dijkstra approximated optimizer...")

        d = optimizer_dijkstra()
        sim_mat = d.compute_similarity(song_points,video_points)

        print("Shrinking the similarity matrix...")  
        sim_mat = cv2.resize(sim_mat,resize_dim)

        w = calc_optim_w(len(song_points),len(video_points))

        song_ids, video_ids = d.dijkstra_combiner(sim_mat,radius=int(w),maxjump=8)

        print("Resizing the results...")

        video_ids = [ int(float(x) * (len(song_points)/resize_dim[1])) for x in video_ids ]    
        video_ids = np.resize(video_ids,len(song_points))

        song_ids = np.array(range(len(song_points)))
        song_ids = [ int(float(x)) for x in song_ids ]    

        print("Looking for mistakes...")

        for i in range(len(song_ids)-1):
            if song_ids[i] == song_ids[i+1]:
                print("Found a mistake in i = ",i)

        print("Done")

        song_points_f = []
        video_points_f = []

        for s_id in song_ids:
            song_points_f.append(song_points[s_id])
        for v_id in video_ids:
            video_points_f.append(video_points[v_id])

        return song_points_f, video_points_f

class optimizer_dtw():

    def optimize(self,song_points,video_points):
        
        print("Running dtw optimizer...")

        song_points.reverse()
        video_points.reverse()

        manhattan_distance = lambda x, y: 1-calc_similarity_sv(x,y)

        w = calc_optim_w(len(song_points),len(video_points))

        d, cost_matrix, acc_cost_matrix, path = dtw.dtw(song_points, video_points, w2=w, dist=manhattan_distance)

        song_points_f = []
        video_points_f = []

        for i in range(len(path[0])-1):
            if (path[0][i] != path[0][i+1]):
                song_points_f.append(song_points[path[0][i]])
                video_points_f.append(video_points[path[1][i]])

        song_points_f.reverse()
        video_points_f.reverse()

        return song_points_f, video_points_f

class sparse_matrix():

    def __init__(self,size,defval):
        self.i_list = []
        self.i_exst = [0]*size
        self.defval = defval
    
    def set_val(self,i,j,val):
        if(self.i_exst[i]==0):
            self.i_list.append([i,[[j,val]]])
            self.i_exst[i]=1
        else:
            for item_i in self.i_list:
                if(item_i[0]==i):
                    for item_j in item_i[1]:
                        if(item_j[0]==j):
                            item_j[1]=val
                            return
                    item_i[1].append([j,val])
                    return

    def get_val(self,i,j):
        if(self.i_exst[i]==0):
            return self.defval
        else:
            for item_i in self.i_list:
                if(item_i[0]==i):
                    for item_j in item_i[1]:
                        if(item_j[0]==j):
                            return item_j[1]
                    return self.defval

    def set_val_v(self,i,v):
        if(self.i_exst[i]==0):
            j_list = v
            self.i_list.append([i,j_list])
            self.i_exst[i]=1
        else:
            for item_i in self.i_list:
                if(item_i[0]==i):
                    for item_j in v:
                        item_i[1].append(item_j)
                    return

    def get_min(self,i):

        min_h = float('inf')
        argmin_h = i-1

        for item_i in self.i_list:
            if(item_i[0]>=0 or item_i[0]<=i-1):
                for item_j in item_i[1]:
                    if(item_j[0]==i):
                        if(item_j[1]<min_h):
                            min_h = item_j[1]
                            argmin_h = item_i[0]

        return min_h, argmin_h

class optimizer_dinprog():
    
    def __init__(self):

        self.p_totspd = 0.01   #Weight for cost of total speedup
        self.p_frmsim = 0.01   #Weight for cost of inter-frame similarity
        self.p_relspd = 0.00   #Weight for cost of relative speedup (not used)
        self.p_emosim = 1.00   #Weight for cost of emotion similarity

    def compute_vsim_p(self,i,F,w,vi):
        i1,i2,k,n = i[0],i[1],i[2],i[3]
        Mat = np.zeros((i2-i1,F))
        #print("  Init part "+str(k)+"/"+str(n))
        for i in range(i1,i2):
            for j in range(max(i-w,0),min(i+w,F)):
                Mat[i-i1,j] = calc_similarity_vv(vi[i],vi[j])
        #print("  Done part "+str(k)+"/"+str(n))
        return Mat
        
    def optimize(self,song_points,video_points,part_id=1):

        M = len(song_points)
        F = len(video_points)
        g = 2
        w = calc_optim_w(M,F)

        C_s = np.zeros((F,F))
        C_r = np.zeros((F,F))
        S_i = np.zeros((F,F))
        S_e = np.zeros((F,M))

        if(self.p_frmsim!=0):
            print("Computing S_i...")
            video_images = []
            for i in range(len(video_points)):
                video_images.append(cv2.resize(cv2.imread(video_points[i][5]),global_cmp_size))
            njobs=num_cores
            ljobs = utils.get_ljobs(0,F,njobs)
            louts = Parallel(n_jobs=njobs)(
                delayed(self.compute_vsim_p)(i,F,3,video_images)
                    for i in ljobs)
            for k in range(len(ljobs)):
                i1,i2 = ljobs[k][0],ljobs[k][1]
                for i in range(i1,i2):
                    for j in range(F):
                        S_i[i][j] = louts[k][i-i1,j]

        if(self.p_totspd!=0):
            v = int(F/M)
            max_cs = 0
            print("Computing C_s...")
            for i in range(F):
                for j in range(F):
                    cs = min(((j-i)-v)**2,200)
                    if(cs>max_cs):
                        max_cs = cs
                    C_s[i,j] = cs
            for i in range(F):
                for j in range(F):
                    C_s[i,j]/=max_cs

        if(self.p_relspd!=0):
            max_cr = 0
            print("Computing C_r...")
            for i in range(F):
                for j in range(F):
                    cr = 1+0.5*(video_points[i][2]+video_points[j][2])
                    if(cr>max_cr):
                        max_cr = cr
                    C_r[i,j] = cr
            for i in range(F):
                for j in range(F):
                    C_r[i,j]/=max_cr

        if(self.p_emosim!=0):
            print("Computing S_e... ")
            for i in range(F):
                for j in range(M):
                    S_e[i,j] = calc_similarity_sv(song_points[j],video_points[i])

        attemp1 = 1
        while(attemp1!=0):

            D = [0]*M
            T = [0]*M
            
            for k in range(M):

                utils.show_progress("Computing Path... ",k,M)  

                D[k] = sparse_matrix(F,float('inf'))
                T[k] = sparse_matrix(F,float('nan'))

                if(k==0):
                    for i in range(F):                                     
                        for j in range(i+1,min(i+1+w,F)):
                            D[k].set_val(i,j,C_s[i,j])
                else:

                    if(attemp1==1):
                        limtmp1 = min(k*w+1,F)
                    elif(attemp1==2):
                        limtmp1 = F

                    for i in range(g,limtmp1):

                        min_h, argmin_h = D[k-1].get_min(i)

                        Dkiv=[]
                        Tkiv=[]
                        
                        for j in range(i+1,min(i+1+w,F)):                 

                            c_totspd = C_s[i,j]
                            c_relspd = C_r[i,j]
                            c_frmsim = 1-S_i[i,j]
                            c_emosim = 1-S_e[j,k]

                            c = self.p_totspd*c_totspd + self.p_frmsim*c_frmsim + self.p_relspd*c_relspd + self.p_emosim*c_emosim

                            Dkiv.append([j,c+min_h])
                            Tkiv.append([j,argmin_h])

                        D[k].set_val_v(i,Dkiv)
                        T[k].set_val_v(i,Tkiv)

            s = F-g
            d = s+1
            min_sd = D[M-1].get_val(s,d)

            for i in range(F-g,F):
                for j in range(i+1,min(i+w,M)):
                    vtmp1 = D[M-1].get_val(i,j)
                    if(vtmp1<min_sd):
                        min_sd = vtmp1
                        s = i
                        d = j
            P = [d]
            k=M

            while(s>g and k>1):
                P.append(s)
                b = T[k-1].get_val(s,d)
                d = s
                s = b
                k = k-1

            P.reverse()

            if(len(P)<=2):
                if(attemp1==1):
                    print("Error: Incorrect Path, retrying...")
                    attemp1=2
                else:
                    print("Error: Incorrect Path, exiting...")
                    attemp1=0
                    exit()
            else:
                attemp1 = 0

            del D
            del T

        video_points_f = []
        song_points_f = []

        while(len(P)<M):
            P.insert(0,0)

        for item in P:
            video_points_f.append(video_points[item])

        song_points_f = song_points
        
        return song_points_f, video_points_f

def get_opt_class_dict():

    opt_class_dict = {
        "uniform": optimizer_uniform,
        #"uniform_p": optimizer_uniform_p,
        #"greedy": optimizer_greedy,
        #"dijkstra": optimizer_dijkstra,
        #"dijkstra_c":optimizer_dijkstra_c,
        #"dtw": optimizer_dtw,
        "ours": optimizer_dinprog,
    }

    return opt_class_dict

def optimize(optimizer,song_points,video_points):

    parallel = True
    if(parallel==True):
        njobs = min(num_cores,max(2,int(len(song_points)/global_par_split_ref)))
        sp_size = int(len(song_points)/njobs)
        vp_size = int(len(video_points)/njobs)

        parts = []

        for i in range(njobs):
            i1 = (i+0)*sp_size
            i2 = (i+1)*sp_size
            if(i==njobs-1):
                i2 = len(song_points)
            sp_part = song_points[i1:i2]
            i1 = (i+0)*vp_size
            i2 = (i+1)*vp_size
            if(i==njobs-1):
                i2 = len(video_points)
            vp_part = video_points[i1:i2]
            parts.append([sp_part,vp_part,i])

        louts = Parallel(n_jobs=njobs)(
            delayed(optimizer.optimize)(part[0],part[1])
                for part in parts)

        song_points_f = []
        video_points_f = []
        for lout in louts:
            for sp in lout[0]:
                song_points_f.append(sp)
            for vp in lout[1]:
                video_points_f.append(vp)
    else:
        song_points_f, video_points_f = optimizer.optimize(song_points,video_points)

    return song_points_f, video_points_f

def pre_accel_video(song_points,video_points):

    song_points_f = []
    video_points_f = []

    nfvi = len(video_points)
    nfvf = min(18000,len(video_points)*0.67)
    spd = nfvi/nfvf

    i = 0
    while(i<nfvi):
        video_points_f.append(video_points[int(i)])
        i+=spd

    nfsi = len(song_points)
    nfsf = min(int(nfvf/3),len(song_points))
    song_points_f = song_points[:nfsf]

    print("Song/video lengths (pre-accel): ",len(song_points_f),len(video_points_f))

    return song_points_f, video_points_f

def run(song_points,video_points,opt_method):

    if(global_quick_test==True):
        song_points = song_points[:100]
        video_points = video_points[:500]

    if(global_preaccel_video==True):
        song_points, video_points = pre_accel_video(song_points,video_points)
    
    if(opt_method=="ours" and len(video_points)>30000):
        video_points2 = []
        for i in range(0,len(video_points),2):
            video_points2.append(video_points[i])
        video_points = video_points2

    opt_class_dict = get_opt_class_dict()
    optimizer1 = opt_class_dict[opt_method]()
    song_points_f,video_points_f =  optimize(optimizer1,song_points,video_points)

    vidsng_diff = len(video_points_f)-len(song_points_f)
    if(vidsng_diff==0):
        print(" >> Video and song sizes already match!")
    elif(vidsng_diff>0):
        print(" >> Video bigger than song, discarding part of video at end")
        video_points_f = video_points_f[0:int(len(song_points_f))]
    else:
        print(" >> Song bigger than video, discarding part of song at end")
        song_points_f = song_points_f[0:int(len(video_points_f))]

    m_sim = evaluator.metrics.calc_emosim(song_points_f,video_points_f)
    print("Mean similarity for "+opt_method+": "+str(m_sim))
        
    return song_points_f, video_points_f
